"""
Configuration management for the Nexla MCP RAG Server.

Loads settings from environment variables with sensible defaults.
All configuration is centralized here to avoid scattered env var reads.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


@dataclass
class Config:
    """Server configuration loaded from environment variables."""

    # LLM settings
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o"))

    # Document settings
    documents_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DOCUMENTS_DIR", "./documents"))
    )
    chroma_db_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CHROMA_DB_DIR", "./data/chroma_db"))
    )

    # Embedding settings
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )

    # Chunking settings
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1500")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "300")))

    # Retrieval settings
    relevance_threshold: float = field(
        default_factory=lambda: float(os.getenv("RELEVANCE_THRESHOLD", "0.3"))
    )

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []

        if self.llm_provider == "openai" and not self.openai_api_key:
            warnings.append(
                "LLM_PROVIDER is set to 'openai' but OPENAI_API_KEY is not set. "
                "Falling back to 'ollama'."
            )
            self.llm_provider = "ollama"

        if self.llm_provider not in ("ollama", "openai"):
            warnings.append(
                f"Unknown LLM_PROVIDER '{self.llm_provider}'. Falling back to 'ollama'."
            )
            self.llm_provider = "ollama"

        if not self.documents_dir.exists():
            warnings.append(
                f"Documents directory '{self.documents_dir}' does not exist. "
                "Creating it now. Place PDFs here and re-ingest."
            )
            self.documents_dir.mkdir(parents=True, exist_ok=True)

        # Ensure ChromaDB directory exists
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)

        return warnings


# Global config singleton
config = Config()
