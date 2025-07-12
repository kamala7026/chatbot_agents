import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import shutil

# --- Configuration ---
# Directory to store uploaded documents and Chroma DB
DATA_DIR = "data"
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploaded_docs")

# Static values for dropdowns
CATEGORIES = ["TGO", "LENS", "AO", "AIC"]
STATUS_OPTIONS = ["Active", "Inactive"]
ACCESS_OPTIONS = ["Internal", "External"]
USER_ROLES = ["Support User", "Non-Support User"]

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Initialize LLM and Embeddings ---
# Initialize Google Gen AI LLM
# Ensure you have your GOOGLE_API_KEY set as an environment variable or pass it directly
# st.secrets["GOOGLE_API_KEY"] assumes it's in .streamlit/secrets.toml
try:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBUA0CTX1rLIk7ecOKUcugUbF5WGth4dt0"
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
except Exception as e:
    st.error(f"Failed to initialize Google Gen AI LLM. Ensure GOOGLE_API_KEY is set. Error: {e}")
    st.stop()

# Initialize HuggingFace Embeddings model
# Using a common sentence-transformer model
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Failed to load HuggingFace Embeddings model. Error: {e}")
    st.stop()

# --- Chroma DB Initialization (Persistent) ---
def get_vector_store():
    """Initializes or loads the Chroma vector store."""
    if not os.path.exists(CHROMA_DB_DIR) or not os.listdir(CHROMA_DB_DIR):
        st.info("Chroma DB not found or empty. It will be initialized upon first document upload.")
        return None # Return None if DB is not initialized yet
    try:
        vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error loading Chroma DB: {e}")
        st.warning("Attempting to re-initialize Chroma DB. You may need to re-upload documents.")
        shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True) # Clean up potentially corrupted DB
        return None

# Global variable to store the vector store instance
vector_store = get_vector_store()

# --- Document Processing Functions ---
def chunk_document(file_path, file_type):
    """
    Loads and chunks a document.
    Uses RecursiveCharacterTextSplitter to simulate section/heading-based chunking.
    For true mixed content (tables, code), a library like 'unstructured' would be ideal,
    but it's not in the allowed tools.
    """
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "txt":
        loader = TextLoader(file_path)
    else:
        st.error(f"Unsupported file type: {file_type}")
        return []

    documents = loader.load()

    # Define separators to simulate section/heading-based chunking
    # This is a simplified approach. For real-world, more robust parsing is needed.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200, # 20% overlap for 1000 chunk size
        separators=[
            "\n\n# ",  # Markdown headings (H1)
            "\n\n## ", # Markdown headings (H2)
            "\n\n### ", # Markdown headings (H3)
            "\n\n",    # Paragraph breaks
            "\n",      # Newlines
            " ",       # Spaces
            ""         # Fallback
        ],
        length_function=len,
        is_separator_regex=False, # Set to True if using regex in separators
    )
    chunks = text_splitter.split_documents(documents)
    st.success(f"Document chunked into {len(chunks)} parts.")
    return chunks

def add_document_to_vector_db(file_path, category, description, status, access):
    """
    Processes a document, chunks it, adds metadata, and adds to Chroma DB.
    """
    global vector_store

    file_name = os.path.basename(file_path)
    file_extension = file_name.split(".")[-1].lower()

    if file_extension not in ["pdf", "txt"]:
        st.error("Only PDF and TXT files are supported.")
        return False

    st.info(f"Processing '{file_name}'...")
    chunks = chunk_document(file_path, file_extension)

    if not chunks:
        st.error("No chunks generated from the document. Skipping vectorization.")
        return False

    # Add metadata to each chunk
    for chunk in chunks:
        chunk.metadata.update({
            "source": file_name,
            "category": category,
            "description": description,
            "status": status,
            "access": access
        })

    # Initialize Chroma DB if it's None
    if vector_store is None:
        st.info(f"Initializing Chroma DB at {CHROMA_DB_DIR}...")
        vector_store = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        st.success("Chroma DB initialized and documents added.")
    else:
        st.info("Adding documents to existing Chroma DB...")
        vector_store.add_documents(chunks)
        st.success("Documents added to Chroma DB.")

    vector_store.persist() # Persist changes
    st.success(f"'{file_name}' processed and added to vector database with metadata.")
    return True

# --- Retrieval and RAG Chain Setup ---

# Define the prompt for the document-based RAG chain
document_qa_prompt = ChatPromptTemplate.from_template(
    """You are an AI assistant specialized in answering questions based on provided document context.
    Answer the question only based on the context provided.
    If the answer is not found in the context, state clearly that you cannot find the answer in the documents.
    Do not make up information.

    Context: {context}

    Question: {input}
    """
)

# Define the prompt for the internet search agent
internet_search_prompt_template = PromptTemplate.from_template(
    """You are an AI assistant that can search the internet to answer questions.
    If the previous attempt to answer from documents failed, use the internet search tool.
    Formulate a concise search query to find the answer.
    Answer the question based on the search results. If you cannot find the answer, state that.

    Question: {input}
    """
)

# --- Tools ---
@tool
def google_search(query: str) -> str:
    """Searches Google for the given query and returns the top results snippet."""
    # In a real application, you would integrate with a proper Google Search API.
    # For this simulation, we'll return a placeholder or use a simple mock.
    # The actual google_search tool is provided by the environment.
    # Assuming `google_search.search` is available from the tool_code environment.
    try:
        search_results = globals()['google_search'].search(queries=[query])
        if search_results and search_results[0].results:
            snippets = [res.snippet for res in search_results[0].results if res.snippet]
            return "\n".join(snippets[:3]) # Return top 3 snippets
        return "No relevant search results found."
    except Exception as e:
        return f"Error during Google Search: {e}"

# --- Main Application Logic ---
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload Documents", "Chatbot"])

    st.title("Document Q&A Chatbot")

    if page == "Upload Documents":
        st.header("Upload New Documents")

        uploaded_file = st.file_uploader("Choose a document (PDF or TXT)", type=["pdf", "txt"])
        document_category = st.selectbox("Category", CATEGORIES)
        document_description = st.text_area("Description", placeholder="Enter a brief description of the document.")
        document_status = st.selectbox("Status", STATUS_OPTIONS)
        document_access = st.selectbox("Access", ACCESS_OPTIONS)

        if st.button("Upload and Process Document"):
            if uploaded_file is not None:
                # Save the uploaded file
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"File '{uploaded_file.name}' saved to {UPLOAD_DIR}")

                # Add to vector DB
                if add_document_to_vector_db(file_path, document_category, document_description, document_status, document_access):
                    st.success("Document successfully uploaded and processed!")
                else:
                    st.error("Failed to process document.")
            else:
                st.warning("Please upload a file first.")

    elif page == "Chatbot":
        st.header("Chat with your Documents")

        if vector_store is None:
            st.warning("No documents loaded yet. Please go to 'Upload Documents' to add files.")
            return

        # User role selection for access control
        user_role = st.sidebar.selectbox("Select Your User Role", USER_ROLES, key="user_role_select")
        st.session_state.user_role = user_role # Store in session state

        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # --- Dynamic Retriever based on User Role ---
                    filter_criteria = {"status": "Active"}
                    if st.session_state.user_role == "Non-Support User":
                        filter_criteria["access"] = "External"
                    # For "Support User", no additional access filter is needed as they get both

                    # Create a retriever with metadata filtering
                    # Chroma's .as_retriever() supports 'where' clause for metadata filtering
                    # Multi-Query Retrieval: Generate multiple reformulations of the original query
                    # This is typically implemented on top of the retriever or as part of the chain.
                    # Langchain's MultiQueryRetriever is a good option.
                    from langchain.retrievers import MultiQueryRetriever

                    base_retriever = vector_store.as_retriever(
                        search_kwargs={"k": 5, "where": filter_criteria}
                    )

                    # Multi-Query Retriever
                    # It generates multiple queries from the original query to improve recall.
                    retriever = MultiQueryRetriever.from_llm(
                        retriever=base_retriever, llm=llm
                    )

                    # --- Langchain Agents and Chains ---

                    # 1. Document-based RAG Chain
                    document_chain = create_stuff_documents_chain(llm, document_qa_prompt)
                    document_rag_chain = create_retrieval_chain(retriever, document_chain)

                    # 2. Internet Search Agent
                    internet_tools = [google_search]
                    internet_agent_prompt = internet_search_prompt_template
                    internet_agent = create_react_agent(llm, internet_tools, internet_agent_prompt)
                    internet_agent_executor = AgentExecutor(agent=internet_agent, tools=internet_tools, verbose=True, handle_parsing_errors=True)


                    # Execute the chain
                    # The input to the chain needs to be a dictionary, e.g., {"input": prompt}
                    # The document_rag_chain returns a dict with 'answer' and 'context'
                    # The internet_agent_executor returns a dict with 'output'
                    try:
                        # First, try the document RAG chain to get its initial answer
                        doc_rag_response = document_rag_chain.invoke({"input": prompt})
                        doc_answer = doc_rag_response.get("answer", "")

                        final_answer = doc_answer
                        # If the document RAG chain couldn't find the answer, then trigger internet search
                        if "cannot find the answer in the documents" in doc_answer.lower() or \
                           "not found in the context" in doc_answer.lower() or \
                           "based on the provided context, I cannot answer" in doc_answer.lower():
                            st.info("Answer not found in documents. Searching the internet...")
                            internet_response = internet_agent_executor.invoke({"input": prompt})
                            final_answer = internet_response.get("output", "Could not find an answer even with internet search.")

                        st.markdown(final_answer)
                        st.session_state.messages.append({"role": "assistant", "content": final_answer})

                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error while processing your request."})

if __name__ == "__main__":
    main()

