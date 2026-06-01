"""
Central configuration loaded from environment variables (or .env).
"""
import os
from dotenv import load_dotenv

from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from project root (three levels up from this file)
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env")
)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="forbid",   # or "ignore" if you prefer
    )

    # Qdrant
    qdrant_url: str
    qdrant_api_key: str
    collection_name: str = "arxiv_papers"

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Retrieval defaults
    top_k_retrieval: int = 20
    top_k_rerank: int = 5

    # Reranker model
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Optional LLM keys – put your real values in .env
    openai_api_key: str = ""
    gemini_api_key: str = ""


def get_settings() -> Settings:
    """Singleton helper for the rest of the app."""
    return Settings()