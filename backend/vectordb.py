import os
import functools
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from backend import config
from backend.logger import get_logger

logger = get_logger(__name__)


def _get_device() -> str:
    """Auto-detect the best available hardware accelerator."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@functools.lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load and cache the sentence-transformer embedding model.

    Model choice: all-MiniLM-L6-v2
    - 384-dimensional dense vectors (config.EMBEDDING_DIM)
    - Runs entirely on CPU — no GPU or API key required
    - Fast enough for real-time indexing of documents up to ~500 pages
    - Used only for candidate retrieval (broad recall); the cross-encoder
      reranker handles precision in the second stage.
    """
    device = _get_device()
    logger.info("[Device] %s running on: %s", config.EMBEDDING_MODEL, device.upper())
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": device},
    )


@functools.lru_cache(maxsize=1)
def get_qdrant_client():
    """Create and cache the local Qdrant client."""
    from qdrant_client import QdrantClient
    os.makedirs(config.QDRANT_PATH, exist_ok=True)
    return QdrantClient(path=config.QDRANT_PATH)


def _ensure_payload_indices(client) -> None:
    """Create keyword payload indices for fast filtered searches.

    Qdrant performs full collection scans when filters are applied without
    indices.  Indexing 'metadata.source' and 'metadata.file_type' — the two
    fields used in every filtered query — converts those scans to O(log n)
    lookups.

    This is idempotent: Qdrant silently skips index creation if it already
    exists, so it is safe to call on every startup.
    """
    from qdrant_client.http import models

    for field in ("metadata.source", "metadata.file_type", "metadata.parent_id", "metadata.user_id"):
        try:
            client.create_payload_index(
                collection_name=config.COLLECTION_NAME,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.debug("[Qdrant] Payload index ensured: %s", field)
        except Exception as exc:
            # Index already exists — this is not an error
            logger.debug("[Qdrant] Index already exists for '%s': %s", field, exc)


@functools.lru_cache(maxsize=1)
def get_vector_store() -> Qdrant:
    """Get or create the Qdrant vector store, with payload indices."""
    embeddings = get_embeddings()
    client = get_qdrant_client()

    try:
        client.get_collection(config.COLLECTION_NAME)
        logger.debug("[Qdrant] Collection '%s' found.", config.COLLECTION_NAME)
    except Exception:
        from qdrant_client.http.models import VectorParams, Distance
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info(
            "[Qdrant] Collection '%s' created (dim=%d).",
            config.COLLECTION_NAME, config.EMBEDDING_DIM,
        )

    _ensure_payload_indices(client)
    return Qdrant(client=client, collection_name=config.COLLECTION_NAME, embeddings=embeddings)


def get_indexed_sources(user_id: str | None = None) -> list:
    """Retrieve all unique 'source' metadata values from Qdrant."""
    try:
        client = get_qdrant_client()
        client.get_collection(config.COLLECTION_NAME)

        records, _ = client.scroll(
            collection_name=config.COLLECTION_NAME,
            scroll_filter=None,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        sources = set()
        for record in records:
            metadata = record.payload.get("metadata", {})
            source = metadata.get("source")
            if source:
                clean_source = source if str(source).startswith("http") else os.path.basename(source)
                if clean_source:
                    sources.add(clean_source)
        return sorted(list(sources))
    except Exception as exc:
        logger.error("[Qdrant] get_indexed_sources failed: %s", exc)
        return []


def delete_source(source_name: str) -> bool:
    """Delete all chunks belonging to a specific source."""
    from qdrant_client.http import models

    try:
        client = get_qdrant_client()
        client.get_collection(config.COLLECTION_NAME)

        records, _ = client.scroll(
            collection_name=config.COLLECTION_NAME,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )

        matching_sources = set()
        for record in records:
            metadata = record.payload.get("metadata", {})
            stored_source = metadata.get("source")
            if not stored_source:
                continue
            stored_source_text = str(stored_source)
            if stored_source_text == source_name or os.path.basename(stored_source_text) == source_name:
                matching_sources.add(stored_source_text)

        if not matching_sources:
            logger.warning("[Qdrant] No indexed chunks found for source: %s", source_name)
            return False

        client.delete(
            collection_name=config.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=source),
                        )
                        for source in matching_sources
                    ]
                )
            ),
            wait=True,
        )

        local_file = os.path.join(config.UPLOAD_DIR, os.path.basename(source_name))
        if os.path.isfile(local_file):
            os.remove(local_file)

        logger.info(
            "[Qdrant] Deleted source '%s' (%d stored source variant(s))",
            source_name, len(matching_sources),
        )
        return True
    except Exception as exc:
        logger.error("[Qdrant] delete_source failed for '%s': %s", source_name, exc)
        return False
