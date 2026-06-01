import os
from pydantic_settings import BaseSettings
from functools import lru_cache

# Resolve the absolute path to the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_FILE = os.path.join(BASE_DIR, ".env")


class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    collection_name: str = "arxiv_papers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_retrieval: int = 20
    top_k_rerank: int = 5

    class Config:
        env_file = ENV_FILE


@lru_cache()
def get_settings() -> Settings:
    return Settings()