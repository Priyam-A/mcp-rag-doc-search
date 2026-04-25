"""
Document ingestion pipeline for the Nexla MCP RAG Server.

Handles PDF parsing (PyMuPDF), page-aware chunking, embedding generation
(sentence-transformers), and ChromaDB indexing. Each chunk carries metadata
for source attribution: document name, page number, and chunk index.
"""

import logging
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass

import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.config import config

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of text extracted from a PDF with source metadata."""

    text: str
    document_name: str
    page_number: int
    chunk_index: int
    section_header: str = ""

    @property
    def chunk_id(self) -> str:
        """Generate a unique, deterministic ID for this chunk."""
        content = f"{self.document_name}:{self.page_number}:{self.chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()

    @property
    def metadata(self) -> dict:
        """Return metadata dict for ChromaDB storage."""
        return {
            "document_name": self.document_name,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "section_header": self.section_header,
        }


class PDFParser:
    """Extracts text from PDFs with page-level metadata using PyMuPDF."""

    @staticmethod
    def extract_pages(pdf_path: Path) -> list[dict]:
        """Extract text and metadata from each page of a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of dicts with 'text', 'page_number', and 'headers' keys.
        """
        pages = []
        try:
            doc = fitz.open(str(pdf_path))
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if not text.strip():
                    continue

                # Detect section headers via font-size heuristics
                headers = PDFParser._detect_headers(page)

                pages.append({
                    "text": text,
                    "page_number": page_num,
                    "headers": headers,
                })
            doc.close()
            logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
        except Exception as e:
            logger.error(f"Failed to parse {pdf_path.name}: {e}")
            raise

        return pages

    @staticmethod
    def _detect_headers(page) -> list[str]:
        """Detect section headers by identifying text with larger font sizes."""
        headers = []
        blocks = page.get_text("dict")["blocks"]

        # Calculate the median font size to use as baseline
        font_sizes = []
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        font_sizes.append(span["size"])

        if not font_sizes:
            return headers

        median_size = sorted(font_sizes)[len(font_sizes) // 2]

        # Text with font size > 1.2x median is considered a header
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text and span["size"] > median_size * 1.2:
                        headers.append(text)

        return headers


class TextChunker:
    """Splits page text into overlapping chunks while tracking page boundaries."""

    def __init__(
        self,
        chunk_size: int = config.chunk_size,
        chunk_overlap: int = config.chunk_overlap,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self, pages: list[dict], document_name: str
    ) -> list[DocumentChunk]:
        """Split document pages into overlapping chunks with metadata.

        Args:
            pages: List of page dicts from PDFParser.extract_pages().
            document_name: Name of the source document.

        Returns:
            List of DocumentChunk objects.
        """
        chunks = []
        chunk_index = 0

        for page in pages:
            text = page["text"]
            page_number = page["page_number"]
            headers = page.get("headers", [])
            current_header = headers[0] if headers else ""

            # Split page text into chunks with overlap
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end].strip()

                if chunk_text:  # Skip empty chunks
                    # Update header context if we pass a header boundary
                    for header in headers:
                        if header in text[start:end]:
                            current_header = header

                    chunks.append(
                        DocumentChunk(
                            text=chunk_text,
                            document_name=document_name,
                            page_number=page_number,
                            chunk_index=chunk_index,
                            section_header=current_header,
                        )
                    )
                    chunk_index += 1

                start += self.chunk_size - self.chunk_overlap

        logger.info(
            f"Created {len(chunks)} chunks from {document_name} "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks


class DocumentStore:
    """Manages ChromaDB vector store for document embeddings."""

    COLLECTION_NAME = "nexla_documents"

    def __init__(
        self,
        db_dir: Path = config.chroma_db_dir,
        embedding_model: str = config.embedding_model,
    ):
        logger.info(f"Initializing DocumentStore at {db_dir}")
        logger.info(f"Loading embedding model: {embedding_model}")

        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        self.embedder = SentenceTransformer(embedding_model)

        # Initialize ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(
            path=str(db_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Nexla RAG document embeddings"},
        )
        logger.info(
            f"ChromaDB collection '{self.COLLECTION_NAME}' ready "
            f"({self.collection.count()} existing chunks)"
        )

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Embed and store document chunks in ChromaDB.

        Args:
            chunks: List of DocumentChunk objects to index.

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedder.encode(texts).tolist()

        # Upsert into ChromaDB (handles duplicates gracefully)
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Indexed {len(chunks)} chunks into ChromaDB")
        return len(chunks)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where_filter: dict | None = None,
    ) -> list[dict]:
        """Query the vector store for relevant chunks.

        Args:
            query_text: Natural language query.
            top_k: Number of results to return.
            where_filter: Optional ChromaDB metadata filter.

        Returns:
            List of result dicts with 'text', 'metadata', and 'score' keys.
        """
        query_embedding = self.embedder.encode(query_text).tolist()

        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_params["where"] = where_filter

        results = self.collection.query(**query_params)

        # Transform ChromaDB results into a cleaner format
        formatted = []
        for i in range(len(results["ids"][0])):
            # ChromaDB returns L2 distances; convert to similarity score (0-1)
            distance = results["distances"][0][i]
            similarity = 1 / (1 + distance)  # Inverse distance as similarity

            formatted.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": round(similarity, 4),
            })

        return formatted

    def delete_document(self, document_name: str) -> int:
        """Delete all chunks belonging to a specific document.

        Args:
            document_name: Name of the document to delete.

        Returns:
            Number of chunks deleted.
        """
        # Get IDs of chunks belonging to this document
        existing = self.collection.get(
            where={"document_name": document_name},
            include=[],
        )
        count = len(existing["ids"])

        if count > 0:
            self.collection.delete(
                where={"document_name": document_name}
            )
            logger.info(f"Deleted {count} chunks for document: {document_name}")
        else:
            logger.warning(f"No chunks found for document: {document_name}")

        return count

    def list_documents(self) -> list[dict]:
        """List all unique documents and their chunk counts.

        Returns:
            List of dicts with document metadata.
        """
        all_data = self.collection.get(include=["metadatas"])

        if not all_data["ids"]:
            return []

        # Aggregate by document name
        doc_info: dict[str, dict] = {}
        for metadata in all_data["metadatas"]:
            name = metadata["document_name"]
            if name not in doc_info:
                doc_info[name] = {
                    "name": name,
                    "chunks": 0,
                    "pages": set(),
                }
            doc_info[name]["chunks"] += 1
            doc_info[name]["pages"].add(metadata["page_number"])

        # Convert sets to counts
        return [
            {
                "name": info["name"],
                "chunks": info["chunks"],
                "pages": len(info["pages"]),
            }
            for info in doc_info.values()
        ]

    def get_total_chunks(self) -> int:
        """Return the total number of chunks in the store."""
        return self.collection.count()


class IngestionPipeline:
    """Orchestrates the full PDF → Chunks → Embeddings → ChromaDB pipeline."""

    def __init__(self, store: DocumentStore | None = None):
        self.parser = PDFParser()
        self.chunker = TextChunker()
        self.store = store or DocumentStore()

    def ingest_pdf(self, pdf_path: Path) -> int:
        """Ingest a single PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Number of chunks indexed.
        """
        logger.info(f"Ingesting: {pdf_path.name}")

        # Parse PDF into pages
        pages = self.parser.extract_pages(pdf_path)

        # Chunk pages
        chunks = self.chunker.chunk_document(pages, pdf_path.name)

        # Embed and store
        count = self.store.add_chunks(chunks)

        logger.info(f"✓ Ingested {pdf_path.name}: {count} chunks from {len(pages)} pages")
        return count

    def ingest_directory(self, directory: Path | None = None) -> dict:
        """Ingest all PDFs from a directory.

        Args:
            directory: Path to directory containing PDFs. Defaults to config.

        Returns:
            Summary dict with ingestion results.
        """
        directory = directory or config.documents_dir

        if not directory.exists():
            raise FileNotFoundError(f"Documents directory not found: {directory}")

        pdf_files = list(directory.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return {
                "status": "no_documents",
                "directory": str(directory),
                "files_found": 0,
            }

        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")

        results = {"status": "success", "directory": str(directory), "files": []}
        total_chunks = 0

        for pdf_path in pdf_files:
            try:
                count = self.ingest_pdf(pdf_path)
                results["files"].append({
                    "name": pdf_path.name,
                    "chunks": count,
                    "status": "success",
                })
                total_chunks += count
            except Exception as e:
                logger.error(f"Failed to ingest {pdf_path.name}: {e}")
                results["files"].append({
                    "name": pdf_path.name,
                    "chunks": 0,
                    "status": f"error: {str(e)}",
                })

        results["total_chunks"] = total_chunks
        results["total_files"] = len(pdf_files)
        logger.info(
            f"✓ Ingestion complete: {total_chunks} chunks from {len(pdf_files)} files"
        )
        return results
