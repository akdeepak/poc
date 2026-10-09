"""Single on-disk vector index shared by the ingestion and search APIs."""
import threading

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import BaseNode

from app.config import INDEX_DIR

_lock = threading.Lock()
_index: VectorStoreIndex | None = None


def get_index() -> VectorStoreIndex | None:
    """Return the index, loading it from disk on first use. None means nothing ingested yet."""
    global _index
    if _index is None and (INDEX_DIR / "docstore.json").exists():
        storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
        _index = load_index_from_storage(storage_context)
    return _index


def add_nodes(nodes: list[BaseNode]) -> None:
    """Embed the chunks, add them to the index and persist it to disk."""
    global _index
    with _lock:
        index = get_index()
        if index is None:
            # Embeddings are generated here using Settings.embed_model
            _index = VectorStoreIndex(nodes)
        else:
            index.insert_nodes(nodes)
        _index.storage_context.persist(persist_dir=str(INDEX_DIR))
