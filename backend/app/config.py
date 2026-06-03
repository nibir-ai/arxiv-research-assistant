from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",          # ignore any unknown keys in .env
    )

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    collection_name: str = "arxiv_papers"

    # Embedding & reranking
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_retrieval: int = 20
    top_k_rerank: int = 5

    # LLM
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma2:2b"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"


@lru_cache()
def get_settings() -> Settings:
    return Settings()