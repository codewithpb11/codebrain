"""Configuration management for CodeBrain."""

from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # LLM - defaults to Ollama (free, local)
    llm_provider: Literal["openai", "anthropic", "ollama"] = "ollama"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-sonnet-20240229"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Embeddings
    embedding_provider: Literal["local", "openai"] = "local"
    embedding_model: str = "all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # Vector Store
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "codebase"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
