import streamlit as st
import pandas as pd
from streamlit_modal import Modal
import logging
import hashlib # Import hashlib

# Import centralized configuration for metadata options
from core.common.config import CATEGORIES, STATUS_OPTIONS, ACCESS_OPTIONS

# Initialize logger
logger = logging.getLogger("aviator_chatbot")

def render_document_management_ui(chatbot, doc_manager):
    """Renders the document management interface."""
    st.header("📋 Manage Documents")

    if not st.session_state.initialized:
        st.warning("⚠️ Please initialize the chatbot first to manage documents.")
        st.info("Steps:\n1. Enter Google API Key in sidebar\n2. Select user type\n3. Click 'Initialize Chatbot'")
        logger.info("Document management UI not rendered: Chatbot not initialized.")
        return
    elif chatbot.vectorstore is None:
        st.error("❌ Vector database not available! Please re-initialize the chatbot.")
        logger.warning("Document management UI: Vector database not available.")
        return

    # Use the passed doc_manager instance
    all_docs = doc_manager.get_all_documents_metadata()

    if not all_docs:
        st.info("No documents found in the database yet. Upload some in the 'Document Upload' tab.")
        logger.info("No documents found for management.")
    else:
        docs_df = pd.DataFrame(all_docs)

        # --- FIX: Ensure 'category' column exists and has no null values ---
        # If the column is missing entirely, create it with a default.
        if 'category' not in docs_df.columns:
            docs_df['category'] = CATEGORIES[0] if CATEGORIES else "General"
        # If the column exists but has null/NaN values, fill them.
        docs_df['category'].fillna(CATEGORIES[0] if CATEGORIES else "General", inplace=True)

        docs_df['Delete?'] = False # Add a column for deletion checkbox
        
        # --- DYNAMIC FIX: Build column config based on the final DataFrame ---
        column_config = {
            "document_id": st.column_config.Column("Document ID", help="Unique ID of the document", disabled=True),
            "filename": st.column_config.Column("File Name", disabled=True),
            "description": st.column_config.Column("Description", disabled=True),
        }

        if 'total_chunks' in docs_df.columns:
            column_config['total_chunks'] = st.column_config.NumberColumn("Chunks", disabled=True)

        if 'access' in docs_df.columns:
            column_config['access'] = st.column_config.SelectboxColumn(
                "Access Level", options=ACCESS_OPTIONS, required=True
            )
            
        if 'status' in docs_df.columns:
            column_config['status'] = st.column_config.SelectboxColumn(
                "Status", options=STATUS_OPTIONS, required=True
            )

        if 'category' in docs_df.columns:
            column_config['category'] = st.column_config.SelectboxColumn(
                "Category", options=CATEGORIES, required=True
            )

        column_config["Delete?"] = st.column_config.CheckboxColumn(
            "Delete?", help="Mark to delete this document", default=False
        )
        
        # --- REVISED FIX: Generate key based on content hash of the DataFrame ---
        # This will only change the key if the data itself changes.
        try:
            # A reliable way to hash a dataframe is to convert it to a byte string first
            data_hash = hashlib.md5(docs_df.to_csv().encode()).hexdigest()
        except Exception as e:
            # Fallback if hashing fails
            logger.warning(f"Failed to hash DataFrame for key, falling back to basic ID: {e}")
            data_hash = str(id(docs_df)) # Use object ID as a fallback

        dynamic_editor_key = f"doc_data_editor_{data_hash}"
        logger.debug(f"Generated data editor key: {dynamic_editor_key}")


        edited_df = st.data_editor(
            docs_df,
            column_config=column_config,
            hide_index=True,
            key=dynamic_editor_key, # <--- CHANGE IS HERE
            num_rows="fixed",
            use_container_width=True
        )

        st.markdown("---")

        delete_modal = Modal(
            "Confirm Deletion",
            key="delete_confirmation_modal",
            padding=20,
            max_width=400
        )

        if st.button(" Apply Changes ", type="primary"):
            logger.info("Apply Changes button clicked in document management.")
            metadata_changes_applied = False
            delete_triggered = False

            # First, check for documents marked for deletion
            docs_to_delete = edited_df[edited_df['Delete?'] == True]
            if not docs_to_delete.empty:
                # For simplicity, handle one deletion at a time. Get the first one.
                doc_to_delete = docs_to_delete.iloc[0]
                st.session_state.pending_delete_doc_id = doc_to_delete['document_id']
                logger.info(f"Delete requested for document ID: {st.session_state.pending_delete_doc_id}. Opening modal.")
                delete_modal.open()
                st.rerun() # Force a rerun to show the modal

            # If no deletions, then process metadata changes
            for original_row, edited_row in zip(docs_df.to_dict(orient='records'),
                                                edited_df.to_dict(orient='records')):
                doc_id_to_process = edited_row['document_id']
                current_doc_updates = {}

                if 'access' in edited_row and original_row['access'] != edited_row['access']:
                    current_doc_updates["access"] = edited_row['access']
                if 'status' in edited_row and original_row['status'] != edited_row['status']:
                    current_doc_updates["status"] = edited_row['status']
                if 'category' in edited_row and original_row['category'] != edited_row['category']:
                    current_doc_updates["category"] = edited_row['category']

                if current_doc_updates:
                    logger.info(f"Detected metadata updates for document {doc_id_to_process}: {current_doc_updates}")
                    with st.spinner(f"Updating fields for {edited_row['filename']}..."):
                        success = doc_manager.update_document_metadata(doc_id_to_process, current_doc_updates)
                        if success:
                            updated_fields_str = ", ".join([f"{field}: **{value}**" for field, value in current_doc_updates.items()])
                            st.success(f"Updated {updated_fields_str} for **{edited_row['filename']}**.")
                            metadata_changes_applied = True
                        else:
                            st.error(f"Failed to update fields for **{edited_row['filename']}**.")
            
            if metadata_changes_applied:
                st.session_state.doc_manager = doc_manager.__class__()
                st.rerun()
            elif not delete_triggered:
                st.info("No changes to apply.")


        # Modal handling logic, now correctly separated
        if delete_modal.is_open():
            with delete_modal.container():
                # Fetch filename for display inside the modal
                doc_to_delete_display = "this document"
                if st.session_state.pending_delete_doc_id:
                     doc_to_delete_display = next((d['filename'] for d in all_docs if d['document_id'] == st.session_state.pending_delete_doc_id), "this document")

                st.write(f"Are you sure you want to permanently delete **{doc_to_delete_display}**?")
                
                col_modal_yes, col_modal_no = st.columns(2)
                with col_modal_yes:
                    if st.button("✅ Yes, Delete", type="primary"):
                        doc_id = str(st.session_state.pending_delete_doc_id) # Ensure it's a string
                        logger.info(f"User confirmed deletion of document ID: {doc_id}")
                        with st.spinner(f"Deleting {doc_to_delete_display}..."):
                            if doc_manager.delete_document(doc_id):
                                st.success(f"🗑️ Document **{doc_to_delete_display}** deleted successfully.")
                                delete_modal.close()
                                st.session_state.doc_manager = doc_manager.__class__()
                                st.session_state.pending_delete_doc_id = None
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to delete {doc_to_delete_display}.")
                                delete_modal.close()
                                st.rerun()

                with col_modal_no:
                    if st.button("❌ No, Cancel"):
                        st.info("Deletion cancelled.")
                        delete_modal.close()
                        st.session_state.pending_delete_doc_id = None
                        logger.info("Document deletion cancelled by user.")
                        st.rerun()

    st.markdown("---")

    if st.button("🔄 Refresh Document List", key="refresh_doc_list_btn"):
        st.cache_data.clear() # Explicitly clear the data cache
        st.session_state.doc_manager = doc_manager.__class__() # Force refresh
        # No explicit key increment needed here, as the new doc_manager will lead to new data_hash
        logger.info("Refresh Document List button clicked.")
        st.rerun()