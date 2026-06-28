import os
import streamlit as st
from backend import config
from backend.logger import get_logger

logger = get_logger(__name__)


def force_remove_readonly(func, path, excinfo):
    """onerror handler for shutil.rmtree — force-removes read-only files on Windows."""
    import stat
    os.chmod(path, stat.S_IWRITE)
    func(path)


def force_close_qdrant():
    """Force-close the local Qdrant connection to release file locks before deletion."""
    try:
        from backend.vectordb import get_vector_store
        store = get_vector_store()
        if store and hasattr(store, "client"):
            store.client.close()
    except Exception:
        pass

    lock_file = os.path.join(config.QDRANT_PATH, ".lock")
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except Exception:
            pass


def perform_clear():
    """Clear all databases and uploaded files, then invalidate all caches."""
    import shutil

    force_close_qdrant()

    cleared = False
    folders_to_clear = [
        config.QDRANT_PATH,
        config.UPLOAD_DIR,
        config.CLONED_REPOS_DIR,
    ]

    for folder in folders_to_clear:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder, onerror=force_remove_readonly)
                # Recreate non-DB directories so uploads don't fail immediately
                if folder != config.QDRANT_PATH:
                    os.makedirs(folder)
                cleared = True
                logger.info("[Utils] Cleared folder: %s", folder)
            except Exception as exc:
                logger.error("[Utils] Failed to clear '%s': %s", folder, exc)
                st.error(f"Failed to clear {folder}: {exc}")

    if cleared:
        from backend.vectordb import get_qdrant_client, get_vector_store, get_embeddings
        from backend.retrieval import get_reranker

        get_qdrant_client.cache_clear()
        get_vector_store.cache_clear()
        get_embeddings.cache_clear()
        get_reranker.cache_clear()

        st.cache_resource.clear()
        st.session_state.messages = []
        st.session_state.pop("confirm_force_delete", None)
        st.rerun()


def has_qdrant_data() -> bool:
    """Lightweight check — does the Qdrant DB exist without loading the client."""
    meta_path = os.path.join(config.QDRANT_PATH, "meta.json")
    return os.path.isfile(meta_path)
