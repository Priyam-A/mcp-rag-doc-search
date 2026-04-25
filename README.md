# 📄 Nexla Document Intelligence MCP Server

An MCP (Model Context Protocol) server that enables AI agents to ask natural-language questions over PDF documents and receive grounded, source-attributed answers via a RAG (Retrieval-Augmented Generation) pipeline.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────┐
│   MCP Client (Claude / Inspector)   │
│         AI Agent sends query        │
└──────────────┬──────────────────────┘
               │ MCP tool call (stdio)
               ▼
┌─────────────────────────────────────┐      ┌─────────────────────────────────────┐
│         FastMCP Server              │      │       FastAPI Companion App         │
│  ┌───────────┬──────────────────┐   │      │       (NotebookLM Frontend)         │
│  │  Tools:   │                  │   │      │                                     │
│  │  • query_documents           │   │◄────►│  • /api/documents (upload/delete)   │
│  │  • list_documents            │   │      │  • /api/chat (Q&A interface)        │
│  │  • get_document_summary      │   │      │  • /api/history (session state)     │
│  │  • compare_documents         │   │      └──────────────────┬──────────────────┘
│  │  • delete_document           │   │                         │
│  │  • ingest_documents          │   │                         │
│  │  • add_document              │   │                         │
│  └───────────┴──────────────────┘   │                         │
└──────────────┬──────────────────────┘                         │
               │                                                │
               ▼                                                │
┌─────────────────────────────────────┐                         │
│          RAG Pipeline               │◄────────────────────────┘
│                                     │
│  Query ──► Embed ──► ChromaDB       │
│                      (retrieve)     │
│                         │           │
│              Context Assembly       │
│              + Source Tracking       │
│                         │           │
│                    LLM (Ollama      │
│                    or OpenAI)       │
│                         │           │
│              Grounded Answer        │
│              + Source Citations      │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│       Document Ingestion            │
│                                     │
│  PDFs ──► PyMuPDF ──► Page-Aware    │
│           Parser      Chunker       │
│                          │          │
│                    Sentence-        │
│                    Transformers     │
│                    (Embeddings)     │
│                          │          │
│                      ChromaDB       │
│                    (Persistent)     │
└─────────────────────────────────────┘
```

### Component Responsibilities

| Layer | File | Responsibility |
|---|---|---|
| **MCP Interface** | `src/server.py` | Tool definitions, MCP protocol compliance |
| **Web API & UI** | `src/api.py`, `public/` | FastAPI companion server and Vanilla JS frontend |
| **RAG Engine** | `src/rag_engine.py` | Query embedding, retrieval, answer generation |
| **Document Ingestion** | `src/ingestion.py` | PDF parsing, chunking, embedding, indexing |
| **LLM Provider** | `src/llm_provider.py` | Abstraction over Ollama / OpenAI |
| **Configuration** | `src/config.py` | Environment variables, defaults, model selection |

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.10+**
- **Ollama** (for local LLM, recommended) — [install here](https://ollama.com/)
- Or an **OpenAI API key** (optional, for cloud LLM)

### Step 1: Clone and Install

```bash
git clone <repo-url>
cd nexla-mcp-rag

# Create virtual environment (requires standard Python 3.11+)
python -m venv venv

# Activate (Windows PowerShell)
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure LLM

**Option A — Ollama (Local, Free, No API Key):**
```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2
```

**Option B — OpenAI (Cloud, Requires API Key):**
```bash
# Copy and edit the environment file
cp .env.example .env
# Add your OpenAI API key to .env
```

### Step 3: Add Documents

Place your PDF files in the `documents/` directory:
```bash
cp /path/to/your/*.pdf documents/
```

### Step 4: Run the Server

You have two ways to interact with the system:

**Option A: The NotebookLM-Inspired Web UI (Bonus)**
We built a custom frontend with chat history, document uploads, and source citations.
```bash
# Start the FastAPI companion server
uvicorn src.api:app --port 8000
# Then open http://localhost:8000 in your browser
```

**Option B: The Raw MCP Server**
To run the standard MCP server on `stdio` transport for Claude Desktop:
```bash
python -m src.server
```

### Step 5: Test with MCP Inspector

```bash
npx -y @modelcontextprotocol/inspector python -m src.server
# Opens at http://127.0.0.1:6274
```

---

## 🔧 MCP Tools

### `query_documents`
Ask a natural language question across all ingested PDF documents.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `question` | `string` | ✅ | — | Natural language question to answer |
| `top_k` | `integer` | ❌ | `5` | Number of relevant chunks to retrieve |

**Example:**
```json
{
  "question": "What is Nexla's approach to data integration?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "Nexla takes a metadata-driven approach to converge diverse integrations...",
  "sources": [
    {"document": "overview.pdf", "page": 2, "section": "Introduction", "relevance_score": 0.89}
  ],
  "query": "What is Nexla's approach to data integration?"
}
```

---

### `list_documents`
List all ingested documents with metadata.

*Takes no parameters.*

**Response:**
```json
{
  "documents": [
    {"name": "overview.pdf", "pages": 12, "chunks": 48, "ingested_at": "2026-04-25T10:00:00"}
  ],
  "total_documents": 4,
  "total_chunks": 192
}
```

---

### `get_document_summary`
Generate a summary of a specific document.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `document_name` | `string` | ✅ | — | Name of the document to summarize |

---

### `compare_documents`
Compare two documents on a specific aspect.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `document_a` | `string` | ✅ | — | First document name |
| `document_b` | `string` | ✅ | — | Second document name |
| `aspect` | `string` | ✅ | — | Aspect to compare (e.g., "methodology") |

---

### `delete_document`
Remove a document and its chunks from the index.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `document_name` | `string` | ✅ | — | Name of the document to delete |

---

### `ingest_documents`
Ingest or re-ingest PDF documents from the documents directory.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `directory` | `string` | ❌ | `./documents` | Path to directory containing PDFs |

---

### `add_document`
Add a single PDF document to the knowledge base and index it immediately.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `source_path` | `string` | ✅ | — | Absolute path to the PDF file to add |

---

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | LLM provider: `ollama` or `openai` |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `OPENAI_API_KEY` | — | OpenAI API key (required if provider is `openai`) |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model name |
| `DOCUMENTS_DIR` | `./documents` | Directory containing PDF files |
| `CHROMA_DB_DIR` | `./data/chroma_db` | ChromaDB persistent storage path |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CHUNK_SIZE` | `1500` | Characters per chunk |
| `CHUNK_OVERLAP` | `300` | Overlap between chunks |
| `RELEVANCE_THRESHOLD` | `0.3` | Minimum similarity score for results |

---

## 🧠 Design Decisions & Trade-offs

### Why PyMuPDF over pdfplumber?
PyMuPDF (C-based engine) is **10-50× faster** than pdfplumber for text extraction. Since we're ingesting 4-5 PDFs at startup, speed matters for developer experience. pdfplumber excels at table extraction, which isn't our primary use case (Q&A over text content).

### Why local embeddings (sentence-transformers) over OpenAI?
**Zero external dependencies for retrieval.** A reviewer can clone and run immediately without configuring an API key. For a small corpus (4-5 PDFs), the quality difference between `all-MiniLM-L6-v2` (384 dims) and OpenAI `text-embedding-3-small` (1536 dims) is negligible.

### Why ChromaDB over FAISS?
ChromaDB provides **built-in persistence and metadata filtering** out of the box. FAISS is faster but requires building custom persistence and metadata layers — unnecessary complexity for 4-5 documents. ChromaDB's `where` filtering lets us filter by `document_name` and `page_number`, enabling source attribution.

### Why dual LLM support (Ollama + OpenAI)?
Ollama provides a **zero-cost, zero-key** experience (ideal for quick evaluation), while OpenAI offers higher answer quality. The config switch lets reviewers choose based on their setup.

### Chunking Strategy
- **Page-boundary-aware splitting:** Each chunk carries `{document_name, page_number, chunk_index}` metadata
- **1500 char chunks with 300 char overlap:** Balances context completeness with retrieval precision
- **Section header detection:** Headings detected via font-size heuristics are included in chunk metadata for finer attribution

### Relevance Threshold
Chunks scoring below `0.3` similarity are discarded before sending to the LLM. This prevents hallucination on off-topic queries — the system returns "I couldn't find relevant information" rather than forcing an answer from low-quality context.

---

## 🤖 Vibe Coding Section

### AI Tools Used
- **Google Antigravity (Gemini)** — Primary AI coding assistant used throughout development

### How AI Was Used
- **Architecture planning:** Discussed technology trade-offs (PyMuPDF vs pdfplumber, ChromaDB vs FAISS, local vs cloud embeddings) with the AI to weigh pros/cons
- **Reference research:** AI searched for FastMCP official documentation and reference implementations (alejandro-ao/RAG-MCP) to understand real-world patterns
- **Code generation:** Used AI for boilerplate generation (project structure, config management, ChromaDB setup)
- **README drafting:** AI helped structure the README to cover all required sections

### What Worked & What Didn't
- **What worked:** Trade-off discussions were highly productive — the AI provided structured comparisons with concrete metrics. Finding and synthesizing documentation from multiple sources (FastMCP docs, ChromaDB API) saved significant research time. The AI was exceptionally good at generating the UI CSS for the NotebookLM aesthetic.
- **What didn't work:** The AI struggled with Windows-specific Python environment debugging. We hit a wall where `pymupdf` failed to install due to a `cmake` compilation error caused by a broken MSYS2 Python environment intercepting commands. The AI kept suggesting generic fixes until I explicitly ordered it to bypass the environment and use absolute paths to a clean Python 3.11 installation. File lock issues (`WinError 32`) on Windows also caused the AI to fail silent cleanup operations. Additionally, during testing, local LLMs (specifically `llama3.2` 3B via Ollama) proved to be rather unstable when handling complex tool-calling via LangChain. The model frequently hallucinated JSON schema structures inside its arguments instead of passing raw strings. While we added explicit prompt engineering and simplified the MCP tool schemas to wrangle Ollama into submission (primarily because we rapidly hit OpenAI rate/token limits during intensive testing and needed a fully local alternative), **using OpenAI (GPT-4) is highly recommended as the significantly more robust and stable approach for production deployments.**

### What I Overrode / Corrected
- Initially the AI suggested using the standalone `fastmcp` package (PrefectHQ). I corrected this to use the **official Anthropic MCP SDK** (`from mcp.server.fastmcp import FastMCP`) for better spec compliance
- Adjusted chunking parameters based on domain knowledge of the document types

### View on AI Tooling in Software Engineering
AI coding assistants excel at **accelerating the known** — boilerplate, API integration patterns, documentation synthesis. They are less reliable for **architectural judgment** — the trade-off between "what's possible" and "what's appropriate" still requires human reasoning. The key skill is knowing when to accept AI suggestions and when to override them with domain knowledge.

---

## 📋 Example Interaction Log

> See [examples/interaction_log.md](examples/interaction_log.md) for full interaction logs with 3+ sample Q&A pairs demonstrating multi-document awareness and source attribution.

---

## 🔮 Future Improvements

With more time, the following enhancements would add value:

- **Streaming Responses:** Stream LangChain agent tool executions and LLM answers back to the Web UI via WebSockets or Server-Sent Events (SSE) for faster perceived response times.
- **Dynamic Retrieval (Dynamic K):** Implement a dynamic `top_k` threshold based on query complexity. Simple queries might only need `top_k=2`, while synthesis queries might dynamically expand to `top_k=15` until the context threshold is met.
- **Dynamic Model Switching:** Add a UI toggle to instantly switch between local inference (Ollama) for privacy and cloud inference (OpenAI) for complex reasoning without restarting the server.
- **Hybrid Search:** Combine semantic (vector) search with BM25 keyword search for better recall on technical terms or proper nouns.
- **Multi-Agent Orchestration:** Expand the LangGraph agent into a multi-agent system (e.g., a "Researcher" agent that queries documents and a "Synthesizer" agent that formats the output).
- **Persistent Agent Memory:** Swap the in-memory chat history for a disk-backed SQLite/Postgres thread database (like LangGraph's `SqliteSaver`) to persist UI sessions across server restarts.
