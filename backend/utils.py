import os
import streamlit as st

def get_indexed_sources_ui():
    """Retrieve indexed sources from the vector db specifically for the UI."""
    from backend.vectordb import get_indexed_sources
    return get_indexed_sources()

def delete_source_ui(source_name):
    """Delete a source from the UI."""
    from backend.vectordb import delete_source
    return delete_source(source_name)

def force_remove_readonly(func, path, excinfo):
    """Force remove readonly files during deletion."""
    import stat
    os.chmod(path, stat.S_IWRITE)
    func(path)

def force_close_qdrant():
    """Force close the local Qdrant connection to release file locks."""
    try:
        from backend.vectordb import get_vector_store
        store = get_vector_store()
        if store and hasattr(store, "client"):
            store.client.close()
    except Exception:
        pass
        
    lock_file = os.path.join("local_qdrant", ".lock")
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except Exception:
            pass

def perform_clear():
    """Clear all databases and uploaded files."""
    import shutil
    
    # Silently force close the Qdrant connection to release file locks before deleting
    force_close_qdrant()
    
    cleared = False
    for folder in ["local_qdrant", "uploaded_docs", "cloned_repos"]:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder, onerror=force_remove_readonly)
                if folder != "local_qdrant":
                    os.makedirs(folder)
                cleared = True
            except Exception as e:
                st.error(f"Failed to clear {folder}: {e}")
                
    if cleared:
        st.cache_resource.clear()
        st.session_state.messages = []
        st.session_state.pop("confirm_force_delete", None)
        st.rerun()

def has_qdrant_data():
    """Lightweight check to see if database exists without heavy imports."""
    meta_path = os.path.join("local_qdrant", "meta.json")
    return os.path.isfile(meta_path)
