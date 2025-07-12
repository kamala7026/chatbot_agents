import os
from typing import List

import PyPDF2
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # Load environment variables from .env file


# Ensure your OpenAI API key is set as an environment variable (e.g., OPENAI_API_KEY)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
google_api_key=""
class AgenticChunker:
    def __init__(self, llm_model_name="gpt-4o-mini", max_chunk_size=1000, chunk_overlap=100):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            temperature=0.7
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ","]
        )
        self.summarize_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert summarizer. Summarize the following text, "
             "focusing on key ideas and maintaining coherence. "
             "Provide a concise title for the summary. output should be below format"
             "Title: Title based on your summarization"
             "Summary: Your summary details"
             ""
             ""),


            ("user", "Text to summarize and title:\n{text}")
        ])
        self.summarize_chain = self.summarize_prompt | self.llm | StrOutputParser()

    def process_document(self, document_path):
        full_text = ""
        try:
            with open(document_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    # --- FIX: Only add text if it's not empty or just whitespace ---
                    if page_text and page_text.strip():
                        # Optionally add page numbers for context, even in this simple setup
                        # text += f"--- Page {page_num + 1} ---\n{page_text.strip()}\n\n"
                        full_text += page_text.strip() + "\n"  # Continue with original style, just stripped
                print(f"Successfully loaded PDF: {document_path}")
            # --- FIX: Return empty list if no meaningful text was extracted ---
        except Exception as e:
            print(f"Error loading PDF {document_path}: {e}")
            raise

        #loader = TextLoader(document_path)
        #raw_documents = loader.load()
        #full_text = raw_documents[0].page_content  # Assuming a single document for simplicity
        processed_chunks = []
        if full_text :
            # Step 1: Initial (recursive) chunking
            initial_chunks = self.text_splitter.split_text(full_text)
            print(f"Initial chunks created: {len(initial_chunks)}")
            for i, chunk in enumerate(initial_chunks):
                print(f"\nProcessing chunk {i + 1}/{len(initial_chunks)}...")
                try:
                    # Step 2: Use LLM to summarize and title each chunk
                    # In a more advanced agentic system, the LLM might also decide if a chunk needs further splitting
                    # or merging based on its content.
                    llm_output = self.summarize_chain.invoke({"text": chunk})

                    # Basic parsing of LLM output for title and summary
                    # This would need more robust parsing in a real application
                    lines = llm_output.split('\n', 1)
                    title = lines[0].replace("Title: ", "").strip() if lines[0].startswith(
                        "Title:") else f"Chunk {i + 1} Summary"
                    summary = lines[1].strip() if len(lines) > 1 else llm_output

                    processed_chunks.append({
                        "original_content": chunk,
                        "title": title,
                        "summary": summary,
                        "chunk_id": f"chunk_{i}"
                    })
                    print(f"  Title: {title}")
                    print(f"  Summary: {summary[:100]}...")  # Print first 100 chars of summary
                except Exception as e:
                    print(f"  Error processing chunk {i}: {e}. Skipping LLM enrichment for this chunk.")
                    processed_chunks.append({
                        "original_content": chunk,
                        "title": f"Chunk {i + 1}",
                        "summary": "Could not generate summary.",
                        "chunk_id": f"chunk_{i}"
                    })
        return processed_chunks

    def process_documents(self, full_text: str):
        processed_chunks = []
        if full_text :
            # Step 1: Initial (recursive) chunking
            initial_chunks = self.text_splitter.split_text(full_text)
            print(f"Initial chunks created: {len(initial_chunks)}")
            for i, chunk in enumerate(initial_chunks):
                print(f"\nProcessing chunk {i + 1}/{len(initial_chunks)}...")
                try:
                    # Step 2: Use LLM to summarize and title each chunk
                    # In a more advanced agentic system, the LLM might also decide if a chunk needs further splitting
                    # or merging based on its content.
                    llm_output = self.summarize_chain.invoke({"text": chunk})

                    # Basic parsing of LLM output for title and summary
                    # This would need more robust parsing in a real application
                    lines = llm_output.split('\n', 1)
                    title = lines[0].replace("Title: ", "").strip() if lines[0].startswith(
                        "Title:") else f"Chunk {i + 1} Summary"
                    summary = lines[1].strip() if len(lines) > 1 else llm_output

                    processed_chunks.append({
                        "original_content": chunk,
                        "title": title,
                        "summary": summary,
                        "chunk_id": f"chunk_{i}"
                    })
                    print(f"  Title: {title}")
                    print(f"  Summary: {summary[:100]}...")  # Print first 100 chars of summary
                except Exception as e:
                    print(f"  Error processing chunk {i}: {e}. Skipping LLM enrichment for this chunk.")
                    processed_chunks.append({
                        "original_content": chunk,
                        "title": f"Chunk {i + 1}",
                        "summary": "Could not generate summary.",
                        "chunk_id": f"chunk_{i}"
                    })
        return processed_chunks

# Create a dummy text file for demonstration
dummy_text = """
The quick brown fox jumps over the lazy dog. This is a classic sentence
used to demonstrate all letters of the alphabet. It's often used in
typing tests and font demonstrations.

Another common phrase is "Pack my box with five dozen liquor jugs."
This sentence also includes every letter of the alphabet. It's a fun
phrase to say and type.

Large Language Models (LLMs) are a type of artificial intelligence
that can understand and generate human-like text. They are trained
on vast amounts of text data. LLMs have revolutionized many areas,
including natural language processing, content creation, and search.

Chunking is a critical preprocessing step for LLMs. It involves
breaking down large documents into smaller, more manageable pieces.
Different chunking strategies exist, such as fixed-size, sentence-based,
and semantic chunking.
"""

with open("sample_document.txt", "w") as f:
    f.write(dummy_text)

# Example usage
if __name__ == "__main__":
    agentic_chunker = AgenticChunker()
    chunks = agentic_chunker.process_document("C:\Application Data\Projects\AI\PythonProject\document_search\ptesting\Story.pdf")

    print("\n--- Final Processed Chunks ---")
    for chunk_data in chunks:
        print(f"Chunk ID: {chunk_data['chunk_id']}")
        print(f"Title: {chunk_data['title']}")
        print(f"Summary: {chunk_data['summary']}")
        print(f"Original Content (first 100 chars): {chunk_data['original_content'][:100]}...")
        print("-" * 30)