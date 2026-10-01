"""Shared embedding model and ChromaDB handle (loaded once per process)."""

import threading

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings

_lock = threading.Lock()
_embeddings: HuggingFaceEmbeddings | None = None
_store: Chroma | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    with _lock:
        if _embeddings is None:
            settings = get_settings()
            query_kwargs: dict = {"normalize_embeddings": True}
            if settings.embedding_query_prefix:
                # BGE models retrieve better when queries carry their instruction prefix.
                query_kwargs["prompt"] = settings.embedding_query_prefix
            _embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                encode_kwargs={"normalize_embeddings": True},
                query_encode_kwargs=query_kwargs,
            )
        return _embeddings


def get_vectorstore() -> Chroma:
    global _store
    embeddings = get_embeddings()
    with _lock:
        if _store is None:
            settings = get_settings()
            settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            _store = Chroma(
                collection_name=settings.chroma_collection,
                embedding_function=embeddings,
                persist_directory=str(settings.chroma_dir),
                collection_metadata={"hnsw:space": "cosine"},
            )
        return _store


def reset_vectorstore() -> Chroma:
    """Drop the policy collection and return a fresh, empty one."""
    global _store
    get_vectorstore().delete_collection()
    with _lock:
        _store = None
    return get_vectorstore()


def chunk_count() -> int:
    return get_vectorstore()._collection.count()
