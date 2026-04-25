import os
import pytest
from pathlib import Path

from src.config import Config
from src.ingestion import DocumentChunk, TextChunker

def test_document_chunk_generation():
    """Test that document chunks generate stable IDs and correct metadata."""
    chunk = DocumentChunk(
        text="This is a test chunk.",
        document_name="test.pdf",
        page_number=1,
        chunk_index=0,
        section_header="Introduction"
    )
    
    # ID should be deterministic
    assert chunk.chunk_id is not None
    assert isinstance(chunk.chunk_id, str)
    
    # Metadata should be formatted correctly for ChromaDB
    meta = chunk.metadata
    assert meta["document_name"] == "test.pdf"
    assert meta["page_number"] == 1
    assert meta["chunk_index"] == 0
    assert meta["section_header"] == "Introduction"

def test_text_chunker():
    """Test that text is split correctly with overlap."""
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    
    # Create dummy page
    pages = [{
        "text": "A" * 45 + "B" * 20, # 65 chars total
        "page_number": 1,
        "headers": []
    }]
    
    chunks = chunker.chunk_document(pages, "test.pdf")
    
    assert len(chunks) == 2
    assert chunks[0].text == "A" * 45 + "B" * 5  # First 50 chars
    # Next chunk starts at 50 - 10 = 40. Length 25.
    assert chunks[1].text == "A" * 5 + "B" * 20
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1

def test_config_validation(monkeypatch):
    """Test that config falls back to ollama if openai key is missing."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "") # Empty key
    
    config = Config()
    warnings = config.validate()
    
    assert len(warnings) > 0
    assert config.llm_provider == "ollama" # Should fallback
