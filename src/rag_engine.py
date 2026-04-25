"""
RAG Engine for the Nexla MCP RAG Server.

Orchestrates the retrieval-augmented generation pipeline:
1. Embed the user's query
2. Retrieve relevant chunks from ChromaDB
3. Filter by relevance threshold
4. Assemble context with source tracking
5. Generate a grounded answer via LLM
"""

import json
import logging

from src.config import config
from src.ingestion import DocumentStore
from src.llm_provider import LLMProvider, create_llm_provider

logger = logging.getLogger(__name__)

# System prompt for grounded Q&A
QA_SYSTEM_PROMPT = """You are a document analysis assistant. Your job is to answer questions
based ONLY on the provided context from PDF documents. Follow these rules:

1. Answer ONLY based on the provided context. Do not use prior knowledge.
2. If the context does not contain enough information, say "I couldn't find sufficient
   information in the documents to answer this question."
3. Be specific and cite which document and page the information comes from.
4. If information comes from multiple documents, synthesize the answer and cite all sources.
5. Keep answers concise but thorough."""

SUMMARY_SYSTEM_PROMPT = """You are a document summarization assistant. Provide a clear,
concise summary of the document content provided. Highlight the key topics, main arguments,
and important details. Keep the summary to 3-5 paragraphs."""

COMPARE_SYSTEM_PROMPT = """You are a document comparison assistant. Compare the two documents
on the specified aspect. Highlight similarities, differences, and key points from each.
Always cite which document each point comes from. Be thorough but concise."""


class RAGEngine:
    """Retrieval-Augmented Generation engine for document Q&A."""

    def __init__(
        self,
        store: DocumentStore | None = None,
        llm: LLMProvider | None = None,
    ):
        self.store = store or DocumentStore()
        self.llm = llm or create_llm_provider(
            provider=config.llm_provider,
            model=(
                config.ollama_model
                if config.llm_provider == "ollama"
                else config.openai_model
            ),
            api_key=config.openai_api_key,
        )
        self.relevance_threshold = config.relevance_threshold

    def query_documents(self, question: str, top_k: int = 5, history: list[dict] | None = None) -> dict:
        """Answer a question using RAG over ingested documents.

        Args:
            question: Natural language question.
            top_k: Number of chunks to retrieve.

        Returns:
            Dict with 'answer', 'sources', and 'query' keys.
        """
        logger.info(f"Query: {question} (top_k={top_k})")

        # Retrieve relevant chunks
        results = self.store.query(query_text=question, top_k=top_k)

        # Filter by relevance threshold
        relevant = [r for r in results if r["score"] >= self.relevance_threshold]

        if not relevant:
            logger.info("No relevant chunks found above threshold")
            return {
                "answer": (
                    "I couldn't find relevant information in the documents "
                    "to answer this question. Try rephrasing or asking about "
                    "a topic covered in the ingested documents."
                ),
                "sources": [],
                "query": question,
            }

        # Assemble context with source citations
        context = self._build_context(relevant)

        # Generate answer via LLM
        prompt = self._build_qa_prompt(question, context)
        answer = self.llm.generate(prompt, system_prompt=QA_SYSTEM_PROMPT, history=history)

        # Build source references
        sources = self._extract_sources(relevant)

        logger.info(
            f"Generated answer from {len(relevant)} chunks "
            f"across {len(set(s['document'] for s in sources))} documents"
        )

        return {
            "answer": answer,
            "sources": sources,
            "query": question,
        }

    def get_document_summary(self, document_name: str) -> dict:
        """Generate a summary of a specific document.

        Args:
            document_name: Name of the document to summarize.

        Returns:
            Dict with 'summary', 'document', and 'chunks_used' keys.
        """
        logger.info(f"Summarizing document: {document_name}")

        # Retrieve all chunks for this document, ordered by chunk_index
        results = self.store.query(
            query_text=f"Summary overview of {document_name}",
            top_k=15,
            where_filter={"document_name": document_name},
        )

        if not results:
            return {
                "summary": f"No content found for document '{document_name}'.",
                "document": document_name,
                "chunks_used": 0,
            }

        context = "\n\n".join(
            f"[Page {r['metadata']['page_number']}]: {r['text']}"
            for r in results
        )

        prompt = (
            f"Summarize the following content from the document '{document_name}':\n\n"
            f"{context}"
        )
        summary = self.llm.generate(prompt, system_prompt=SUMMARY_SYSTEM_PROMPT)

        return {
            "summary": summary,
            "document": document_name,
            "chunks_used": len(results),
        }

    def compare_documents(
        self, document_a: str, document_b: str, aspect: str
    ) -> dict:
        """Compare two documents on a specific aspect.

        Args:
            document_a: First document name.
            document_b: Second document name.
            aspect: Aspect to compare (e.g., "methodology", "conclusions").

        Returns:
            Dict with 'comparison', 'documents', and 'aspect' keys.
        """
        logger.info(f"Comparing {document_a} vs {document_b} on: {aspect}")

        # Retrieve relevant chunks from each document
        results_a = self.store.query(
            query_text=aspect,
            top_k=5,
            where_filter={"document_name": document_a},
        )
        results_b = self.store.query(
            query_text=aspect,
            top_k=5,
            where_filter={"document_name": document_b},
        )

        if not results_a and not results_b:
            return {
                "comparison": (
                    f"No relevant content found in either document for the aspect '{aspect}'."
                ),
                "documents": [document_a, document_b],
                "aspect": aspect,
            }

        context_a = "\n".join(
            f"[Page {r['metadata']['page_number']}]: {r['text']}"
            for r in results_a
        )
        context_b = "\n".join(
            f"[Page {r['metadata']['page_number']}]: {r['text']}"
            for r in results_b
        )

        prompt = (
            f"Compare the following two documents on the aspect: '{aspect}'\n\n"
            f"=== Document A: {document_a} ===\n{context_a}\n\n"
            f"=== Document B: {document_b} ===\n{context_b}"
        )
        comparison = self.llm.generate(prompt, system_prompt=COMPARE_SYSTEM_PROMPT)

        return {
            "comparison": comparison,
            "documents": [document_a, document_b],
            "aspect": aspect,
        }

    def _build_context(self, results: list[dict]) -> str:
        """Build a context string from retrieved chunks with source markers."""
        parts = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            source = (
                f"[Source {i}: {meta['document_name']}, "
                f"Page {meta['page_number']}"
            )
            if meta.get("section_header"):
                source += f", Section: {meta['section_header']}"
            source += f", Relevance: {r['score']:.2f}]"

            parts.append(f"{source}\n{r['text']}")

        return "\n\n---\n\n".join(parts)

    def _build_qa_prompt(self, question: str, context: str) -> str:
        """Build the LLM prompt for Q&A."""
        return (
            f"Answer the following question based on the provided document context.\n\n"
            f"Question: {question}\n\n"
            f"Document Context:\n{context}\n\n"
            f"Provide a clear, grounded answer with references to the source documents."
        )

    def _extract_sources(self, results: list[dict]) -> list[dict]:
        """Extract deduplicated source references from results."""
        sources = []
        seen = set()

        for r in results:
            meta = r["metadata"]
            key = (meta["document_name"], meta["page_number"])
            if key not in seen:
                seen.add(key)
                sources.append({
                    "document": meta["document_name"],
                    "page": meta["page_number"],
                    "section": meta.get("section_header", ""),
                    "relevance_score": r["score"],
                })

        return sources
