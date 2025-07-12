import streamlit as st
import pandas as pd
from streamlit_modal import Modal
import logging
import hashlib # Import hashlib

# Initialize logger
logger = logging.getLogger("aviator_chatbot")

# Constants (consistent with core/chatbot.py)
CATEGORIES = ["TGO", "LENS", "AO", "AIC"]
STATUS_OPTIONS = ["Active", "Inactive"]
ACCESS_OPTIONS = ["Internal", "External"]

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
        docs_df['Delete?'] = False # Add a column for deletion checkbox

        column_config = {
            "document_id": st.column_config.Column("Document ID", help="Unique ID of the document",
                                                   disabled=True),
            "filename": st.column_config.Column("File Name", disabled=True),
            "description": st.column_config.Column("Description", disabled=True),
            "total_chunks": st.column_config.NumberColumn("Chunks", disabled=True),
            "access": st.column_config.SelectboxColumn(
                "Access Level",
                help="Change access for this document",
                options=ACCESS_OPTIONS,
                required=True
            ),
            "status": st.column_config.SelectboxColumn(
                "Status",
                help="Change status for this document (Active/Inactive)",
                options=STATUS_OPTIONS,
                required=True
            ),
            "category": st.column_config.SelectboxColumn(
                "Category",
                help="Change category for this document",
                options=CATEGORIES,
                required=True
            ),
            "Delete?": st.column_config.CheckboxColumn(
                "Delete?",
                help="Mark to delete this document",
                default=False
            ),
            "key": st.column_config.Column("Key", width="hidden") # Hide internal key if exists
        }

        # --- REVISED FIX: Generate key based on content hash of the DataFrame ---
        # This will only change the key if the data itself changes.
        try:
            # Use pandas' built-in hash_pandas_object for robust hashing
            data_hash = hashlib.md5(pd.util.hash_pandas_object(docs_df, index=True).values).hexdigest()
        except Exception as e:
            # Fallback if hashing fails (e.g., empty DataFrame in some edge cases)
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

            for original_row, edited_row in zip(docs_df.to_dict(orient='records'),
                                                edited_df.to_dict(orient='records')):
                doc_id_to_process = edited_row['document_id']

                current_doc_updates = {}
                if original_row['access'] != edited_row['access']:
                    current_doc_updates["access"] = edited_row['access']
                if original_row['status'] != edited_row['status']:
                    current_doc_updates["status"] = edited_row['status']
                if original_row['category'] != edited_row['category']:
                    current_doc_updates["category"] = edited_row['category']

                if current_doc_updates:
                    logger.info(f"Detected metadata updates for document {doc_id_to_process}: {current_doc_updates}")
                    with st.spinner(f"Updating fields for {edited_row['filename']}..."):
                        success = doc_manager.update_document_metadata(
                            doc_id_to_process,
                            current_doc_updates
                        )
                        if success:
                            updated_fields_str = ", ".join(
                                [f"{field}: **{value}**" for field, value in current_doc_updates.items()])
                            st.success(f"Updated {updated_fields_str} for **{edited_row['filename']}**.")
                            metadata_changes_applied = True
                            logger.info(f"Successfully updated metadata for {edited_row['filename']}.")
                        else:
                            st.error(f"Failed to update fields for **{edited_row['filename']}**.")
                            logger.error(f"Failed to update metadata for {edited_row['filename']}.")

                if edited_row['Delete?'] and not original_row['Delete?']:
                    st.session_state.pending_delete_doc_id = doc_id_to_process
                    delete_triggered = True
                    logger.info(f"Delete requested for document ID: {doc_id_to_process}. Opening modal.")
                    break # Break to handle modal, then rerun

            if delete_triggered:
                delete_modal.open()

            # Modal handling logic (remains the same)
            if delete_modal.is_open():
                with delete_modal.container():
                    doc_to_delete_display = next(
                        (d['filename'] for d in doc_manager.get_all_documents_metadata() if
                         d['document_id'] == st.session_state.pending_delete_doc_id), "this document")
                    st.write(
                        f"Are you sure you want to permanently delete **{doc_to_delete_display}** (ID: {st.session_state.pending_delete_doc_id[-6:]})?")
                    col_modal_yes, col_modal_no = st.columns(2)

                    with col_modal_yes:
                        if st.button("✅ Yes, Delete", key="modal_confirm_delete_yes", type="secondary"):
                            logger.info(f"User confirmed deletion of document ID: {st.session_state.pending_delete_doc_id}")
                            with st.spinner(f"Deleting {doc_to_delete_display}..."):
                                if doc_manager.delete_document(st.session_state.pending_delete_doc_id):
                                    st.success(f"🗑️ Document **{doc_to_delete_display}** deleted successfully.")
                                    delete_modal.close()
                                    st.session_state.doc_manager = doc_manager.__class__() # Refresh manager
                                    st.session_state.pending_delete_doc_id = None
                                    # No explicit key increment needed here, as the new doc_manager will lead to new data_hash
                                    logger.info(f"Document {doc_to_delete_display} successfully deleted.")
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to delete {doc_to_delete_display}.")
                                    delete_modal.close()
                                    st.session_state.pending_delete_doc_id = None
                                    logger.error(f"Failed to delete document {doc_to_delete_display}.")
                                    st.rerun()
                    with col_modal_no:
                        if st.button("❌ No, Cancel", key="modal_confirm_delete_no"):
                            st.info("Deletion cancelled.")
                            delete_modal.close()
                            st.session_state.pending_delete_doc_id = None
                            logger.info("Document deletion cancelled by user.")
                            st.rerun()

            elif metadata_changes_applied:
                st.session_state.doc_manager = doc_manager.__class__() # Refresh manager to reflect changes
                # No explicit key increment needed here, as the new doc_manager will lead to new data_hash
                st.rerun()
            else:
                st.info("No changes to apply or no documents marked for deletion.")
                logger.info("No changes or deletions triggered by 'Apply Changes' button.")

    st.markdown("---")

    if st.button("🔄 Refresh Document List", key="refresh_doc_list_btn"):
        st.session_state.doc_manager = doc_manager.__class__() # Force refresh
        # No explicit key increment needed here, as the new doc_manager will lead to new data_hash
        logger.info("Refresh Document List button clicked.")
        st.rerun()