# 📄 Document Intelligence MCP Server

An MCP (Model Context Protocol) server that enables AI agents to ask natural-language questions over PDF documents and receive grounded, source-attributed answers via a RAG (Retrieval-Augmented Generation) pipeline. Includes a **bonus** NotebookLM-inspired Web UI powered by an autonomous LangChain agent that consumes the MCP server over stdio.

## 🏗️ Architecture Overview

This project has a **dual-interface architecture**: a standards-compliant MCP Server for external AI clients, and a Web UI powered by an autonomous LangChain agent that consumes that same MCP Server over stdio.

```
                      ┌──────────────────────────────────────┐
                      │      Web UI (Browser)                │
                      │      NotebookLM-style Frontend       │
                      └──────────────┬───────────────────────┘
                                     │ HTTP (/api/chat)
                                     ▼
                      ┌──────────────────────────────────────┐
                      │      FastAPI Server (src/api.py)     │
                      │                                      │
                      │   ┌──────────────────────────────┐   │
                      │   │  LangGraph ReAct Agent       │   │
                      │   │  (src/agent.py)              │   │
                      │   │  Autonomous tool selection   │   │
                      │   └──────────────┬───────────────┘   │
                      └──────────────────┼───────────────────┘
                                         │ langchain-mcp-adapters
                                         │ (stdio transport)
┌────────────────┐                       ▼
│  MCP Client    │    stdio    ┌──────────────────────────────────────┐
│  (Claude /     │────────────►│      FastMCP Server (src/server.py) │
│   Inspector)   │             │                                      │
└────────────────┘             │  Tools:                              │
                               │   • query_documents                  │
                               │   • list_documents                   │
                               │   • get_document_summary             │
                               │   • compare_documents                │
                               │   • delete_document                  │
                               │   • add_document                     │
                               │   • ingest_documents                 │
                               └──────────────┬───────────────────────┘
                                              │
                                              ▼
                               ┌──────────────────────────────────────┐
                               │      RAG Pipeline                    │
                               │                                      │
                               │  Query ──► Embed ──► ChromaDB        │
                               │                      (retrieve)      │
                               │                         │            │
                               │              Context Assembly        │
                               │              + Source Tracking        │
                               │                         │            │
                               │                    LLM (Ollama       │
                               │                    or OpenAI)        │
                               │                         │            │
                               │              Grounded Answer         │
                               │              + Source Citations       │
                               └──────────────────────────────────────┘
```

**Key insight:** The Web UI does _not_ call RAG functions directly. Instead, the LangChain agent spawns the MCP server as a subprocess and communicates with it over the official MCP protocol using `langchain-mcp-adapters`. This means both Claude Desktop and the Web UI consume the **exact same tool interface**.

### Component Responsibilities

| Layer | File | Responsibility |
|---|---|---|
| **MCP Server** | `src/server.py` | Tool definitions, MCP protocol compliance via FastMCP |
| **LangChain Agent** | `src/agent.py` | Autonomous tool selection via LangGraph ReAct agent |
| **Web API & UI** | `src/api.py`, `public/` | FastAPI server, MCP client lifecycle, chat endpoints |
| **RAG Engine** | `src/rag_engine.py` | Query embedding, retrieval, LLM answer generation |
| **Document Ingestion** | `src/ingestion.py` | PDF parsing, chunking, embedding, ChromaDB indexing |
| **LLM Provider** | `src/llm_provider.py` | Abstraction over Ollama / OpenAI for the RAG pipeline |
| **Configuration** | `src/config.py` | Environment variables, defaults, validation |

### LangChain + MCP Integration (Bonus)

Rather than having the Web UI call backend Python functions directly, we took an intentionally over-engineered approach to demonstrate full MCP understanding:

1. **On server startup**, `src/api.py` uses `langchain-mcp-adapters` to launch `python -m src.server` as a background subprocess.
2. The adapter connects to the subprocess over **stdio** and discovers all available MCP tools automatically.
3. Those tools are converted into native LangChain tools and handed to a **LangGraph ReAct agent**.
4. When a user sends a chat message, the agent **autonomously decides** which MCP tool to call (query, summarize, compare, etc.) and synthesizes the result into a natural language response.

This proves the MCP server is not just a standalone artifact — it is a fully consumable, protocol-compliant service that any MCP client (Claude, the Inspector, or our own LangChain agent) can use interchangeably.

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.10+**
- **Ollama** (for local LLM, recommended) — [install here](https://ollama.com/)
- Or an **OpenAI API key** (optional, for cloud LLM)

### Step 1: Clone and Install

```bash
git clone https://github.com/Priyam-A/mcp-rag-doc-search.git
cd mcp-rag-doc-search

# Create virtual environment
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
# Create a .env file in the project root with:
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

### Step 3: Add Documents

The provided PDF files are already included in the `documents/` directory. To add your own:
```bash
cp /path/to/your/*.pdf documents/
```

### Step 4: Run the Server

You have two ways to interact with the system:

**Option A: The NotebookLM-Inspired Web UI (Recommended)**
We built a custom frontend with chat history, document uploads, and source citations.
```bash
# Start the FastAPI companion server
uvicorn src.api:app --port 8000
# Then open http://localhost:8000 in your browser
```

**Option B: The Raw MCP Server**
To run the standard MCP server on `stdio` transport for Claude Desktop or other MCP clients:
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

### Why LangChain + MCP Adapters for the Web UI?
We intentionally chose to have the Web UI consume the MCP server over stdio (via `langchain-mcp-adapters`) rather than calling Python functions directly. This proves the MCP server is a fully standalone, protocol-compliant service — not just a code library. The LangGraph ReAct agent adds autonomous tool selection, so the UI doesn't need to hardcode which tool to call for each query.

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

**Architecture & Design Phase:**
- Prompted the AI to compare technology options with structured trade-off tables (e.g., "Compare PyMuPDF vs pdfplumber for PDF text extraction — focus on speed, table support, and dependency complexity"). The AI returned detailed comparisons with concrete metrics that directly informed my choices.
- Asked the AI to search the web for `langchain mcp` after I decided I wanted the Web UI to communicate with the MCP server over the official protocol. The AI found the `langchain-mcp-adapters` package, read its documentation, and designed the integration architecture using `MultiServerMCPClient` with stdio transport. This was a pivotal discovery that elevated the project from "MCP server + separate web app" to "MCP server consumed by its own LangChain agent."

**Implementation Phase:**
- Used AI for boilerplate generation: project scaffolding, dataclass definitions, ChromaDB client setup, FastAPI CORS configuration.
- The AI generated the full NotebookLM-inspired frontend CSS from a single prompt describing the aesthetic I wanted. The glassmorphism effects and dark theme were produced on the first attempt.
- Reference research: AI searched for FastMCP official documentation and reference implementations to understand real-world MCP patterns.

**Debugging Phase:**
- Spent significant time debugging Ollama's tool-calling instability with the AI. The local `llama3.2` 3B model repeatedly hallucinated JSON schema structures inside tool arguments (e.g., passing `{"type": "string", "value": "my question"}` instead of `"my question"`). The AI and I iterated through multiple fixes: simplifying Pydantic `Annotated` wrappers, adding few-shot examples to the system prompt, and upgrading to `qwen2.5:7b`. This debugging cycle taught me the practical limits of small local models for agentic workflows.

### What Worked & What Didn't
- **What worked:** Trade-off discussions were highly productive — the AI provided structured comparisons with concrete metrics. Finding and synthesizing documentation from multiple sources (FastMCP docs, ChromaDB API, LangChain MCP adapters) saved significant research time. The AI was exceptionally good at generating the UI CSS for the NotebookLM aesthetic.
- **What didn't work:** The AI struggled with Windows-specific Python environment debugging. We hit a wall where `pymupdf` failed to install due to a `cmake` compilation error caused by a broken MSYS2 Python environment intercepting commands. The AI kept suggesting generic fixes until I explicitly ordered it to bypass the environment and use absolute paths to a clean Python 3.11 installation. File lock issues (`WinError 32`) on Windows also caused the AI to fail silent cleanup operations. Additionally, during testing, local LLMs (specifically `llama3.2` 3B via Ollama) proved to be rather unstable when handling complex tool-calling via LangChain. The model frequently hallucinated JSON schema structures inside its arguments instead of passing raw strings. While we added explicit prompt engineering and simplified the MCP tool schemas to wrangle Ollama into submission (primarily because we rapidly hit OpenAI rate/token limits during intensive testing and needed a fully local alternative), **using OpenAI (GPT-4) is highly recommended as the significantly more robust and stable approach for production deployments.**

### What I Overrode / Corrected
- Initially the AI suggested using the standalone `fastmcp` package (PrefectHQ). I corrected this to use the **official Anthropic MCP SDK** (`from mcp.server.fastmcp import FastMCP`) for better spec compliance
- Adjusted chunking parameters based on domain knowledge of the document types
- Corrected multiple `langgraph` API breaking changes (`state_modifier` → `messages_modifier` → `prompt`) that the AI initially got wrong due to stale training data

### View on AI Tooling in Software Engineering
AI coding assistants excel at **accelerating the known** — boilerplate, API integration patterns, documentation synthesis. They are less reliable for **architectural judgment** — the trade-off between "what's possible" and "what's appropriate" still requires human reasoning. They are particularly unreliable when APIs evolve rapidly (as we saw with LangGraph's parameter naming changes across versions). The key skill is knowing when to accept AI suggestions and when to override them with domain knowledge.

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
