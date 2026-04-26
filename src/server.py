"""
Nexla Document Intelligence MCP Server.

Exposes document Q&A tools via the Model Context Protocol (MCP).
Uses the official Anthropic MCP SDK with FastMCP for clean tool definitions.

Tools:
    - query_documents: Ask questions across all ingested PDFs
    - list_documents: List all ingested documents with metadata
    - get_document_summary: Generate a summary of a specific document
    - compare_documents: Compare two documents on a specific aspect
    - delete_document: Remove a document from the index
    - ingest_documents: Ingest/re-ingest PDFs from a directory
"""

import json
import logging
import sys
import shutil
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.config import config
from src.ingestion import DocumentStore, IngestionPipeline
from src.rag_engine import RAGEngine

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,  # MCP uses stdout for protocol; logs go to stderr
)
logger = logging.getLogger("nexla-mcp")

# --- Initialize MCP Server ---
mcp = FastMCP(
    "Nexla Document Intelligence",
    instructions=(
        "This server provides document Q&A tools over ingested PDF documents. "
        "Use 'list_documents' to see available documents, then 'query_documents' "
        "to ask questions. Every answer includes source attribution."
    ),
)

# --- Initialize Components ---
logger.info("Initializing Nexla MCP RAG Server...")

# Validate config
warnings = config.validate()
for w in warnings:
    logger.warning(w)

# Initialize document store and RAG engine
store = DocumentStore()
pipeline = IngestionPipeline(store=store)
engine = RAGEngine(store=store)

# Auto-ingest documents at startup
try:
    if config.documents_dir.exists() and list(config.documents_dir.glob("*.pdf")):
        logger.info(f"Auto-ingesting PDFs from {config.documents_dir}...")
        result = pipeline.ingest_directory()
        logger.info(
            f"Startup ingestion: {result.get('total_chunks', 0)} chunks "
            f"from {result.get('total_files', 0)} files"
        )
    else:
        logger.info(
            f"No PDFs found in {config.documents_dir}. "
            "Use the 'ingest_documents' tool to add documents."
        )
except Exception as e:
    logger.error(f"Startup ingestion failed: {e}")


# --- MCP Tool Definitions ---


@mcp.tool()
def query_documents(question: str, top_k: int = 5) -> str:
    """Ask a natural language question across all ingested PDF documents.

    Returns a grounded answer with source attribution including document name,
    page number, and relevance score. Supports multi-document queries.
    """
    try:
        result = engine.query_documents(question=question, top_k=top_k)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"query_documents error: {e}")
        return json.dumps({
            "error": str(e),
            "query": question,
        })


@mcp.tool()
def list_documents() -> str:
    """List all ingested documents with metadata (name, page count, chunk count).

    Use this to discover what documents are available before querying.
    """
    try:
        documents = store.list_documents()
        return json.dumps({
            "documents": documents,
            "total_documents": len(documents),
            "total_chunks": store.get_total_chunks(),
        }, indent=2)
    except Exception as e:
        logger.error(f"list_documents error: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_document_summary(document_name: str) -> str:
    """Generate a concise summary of a specific document.

    Retrieves key content from the document and produces a 3-5 paragraph summary
    highlighting main topics, arguments, and important details.
    """
    try:
        result = engine.get_document_summary(document_name=document_name)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"get_document_summary error: {e}")
        return json.dumps({"error": str(e), "document": document_name})


@mcp.tool()
def compare_documents(document_a: str, document_b: str, aspect: str) -> str:
    """Compare two documents on a specific aspect.

    Retrieves relevant content from both documents and generates a structured
    comparison highlighting similarities, differences, and key points.
    Demonstrates multi-document awareness.
    """
    try:
        result = engine.compare_documents(
            document_a=document_a,
            document_b=document_b,
            aspect=aspect,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"compare_documents error: {e}")
        return json.dumps({
            "error": str(e),
            "documents": [document_a, document_b],
            "aspect": aspect,
        })


@mcp.tool()
def delete_document(document_name: str) -> str:
    """Remove a document and all its chunks from the index.

    This permanently deletes the document's embeddings from the vector store.
    The original PDF file is not deleted from disk.
    """
    try:
        count = store.delete_document(document_name=document_name)
        if count > 0:
            return json.dumps({
                "status": "deleted",
                "document": document_name,
                "chunks_removed": count,
            }, indent=2)
        else:
            return json.dumps({
                "status": "not_found",
                "document": document_name,
                "message": f"No document named '{document_name}' found in the index.",
            }, indent=2)
    except Exception as e:
        logger.error(f"delete_document error: {e}")
        return json.dumps({"error": str(e), "document": document_name})


@mcp.tool()
def add_document(source_path: str) -> str:
    """Add a single PDF document to the knowledge base and index it immediately.

    Copies the provided PDF file into the local documents directory and runs the
    ingestion pipeline on it so it becomes available for queries.
    """
    try:
        source_file = Path(source_path)
        if not source_file.exists() or not source_file.is_file():
            return json.dumps({
                "error": "File not found or is not a valid file",
                "source_path": source_path
            }, indent=2)
            
        if source_file.suffix.lower() != ".pdf":
            return json.dumps({
                "error": "Only PDF files are supported",
                "source_path": source_path
            }, indent=2)
            
        # Copy file to documents directory
        config.documents_dir.mkdir(parents=True, exist_ok=True)
        target_path = config.documents_dir / source_file.name
        
        shutil.copy2(source_file, target_path)
        
        # Ingest just this new file
        chunks_added = pipeline.ingest_pdf(target_path)
        
        return json.dumps({
            "status": "success",
            "document": source_file.name,
            "chunks_added": chunks_added,
            "message": f"Successfully added and indexed {source_file.name}"
        }, indent=2)
    except Exception as e:
        logger.error(f"add_document error: {e}")
        return json.dumps({"error": str(e), "source_path": source_path})


@mcp.tool()
def ingest_documents(directory: str = "./documents") -> str:
    """Ingest or re-ingest PDF documents from a directory.

    Parses all PDFs in the specified directory, splits them into chunks,
    generates embeddings, and indexes them in the vector store.
    Existing chunks from the same documents are updated (upserted).
    """
    try:
        dir_path = Path(directory)
        result = pipeline.ingest_directory(directory=dir_path)
        return json.dumps(result, indent=2)
    except FileNotFoundError as e:
        return json.dumps({
            "error": str(e),
            "directory": directory,
            "suggestion": "Create the directory and place PDF files in it.",
        }, indent=2)
    except Exception as e:
        logger.error(f"ingest_documents error: {e}")
        return json.dumps({"error": str(e), "directory": directory})


# --- Entry Point ---

if __name__ == "__main__":
    logger.info("Starting Nexla MCP server on stdio transport...")
    mcp.run()
