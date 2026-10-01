from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Secrets / connections (from .env)
    groq_api_key: str = ""
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "policy_assistant"
    mongodb_collection: str = "chat_history"

    # LLM
    groq_model: str = "openai/gpt-oss-120b"
    groq_rewrite_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.0

    # Embeddings / vector store
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    chroma_dir: Path = BACKEND_DIR / "chroma_db"
    chroma_collection: str = "company_policy"
    retrieval_k: int = 7

    # Chunking
    chunk_size: int = 900
    chunk_overlap: int = 120

    # Policy source & metadata defaults (used when not detectable in the document)
    policy_path: Path = BACKEND_DIR / "data" / "policy" / "code_of_conduct_v2.txt"
    policy_name: str = "Code of Conduct & Ethics"
    policy_version: str = "Version 2"
    policy_effective_date: str = "01 January 2026"
    whistleblower_url: str = "https://nazar.indianic.biz/"

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
