# Example Interaction Log

> These are sample Q&A interactions demonstrating the MCP server's capabilities.
> All queries were run against the provided PDF documents using `query_documents`, `compare_documents`, and `get_document_summary` tools.

---

## Example 1: Single Document Query (Source Attribution)

**Question:** "What are the key responsibilities of the Software Engineer role at Nexla?"

**Tool Called:** `query_documents`

```json
{
  "question": "What are the key responsibilities of the Software Engineer role at Nexla?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "The Software Engineer role at Nexla involves embedding directly with strategic customers, understanding their real-world data challenges, and shipping production-grade solutions on top of the Nexla platform. Key responsibilities include building and maintaining integration pipelines, working with AI/ML workflows, and operating as a 'technical co-founder' for each customer engagement. The role requires the ability to own an outcome, reason under ambiguity, and communicate decisions clearly.",
  "sources": [
    {
      "document": "JD_Software Engineeer (1).pdf",
      "page": 1,
      "section": "Role Description",
      "relevance_score": 0.92
    },
    {
      "document": "Nexla_Software Engineer_Assignment.pdf",
      "page": 4,
      "section": "A Note from the Team",
      "relevance_score": 0.78
    }
  ],
  "query": "What are the key responsibilities of the Software Engineer role at Nexla?"
}
```

**Demonstrates:** Multi-document awareness — the answer pulls context from both the job description and the assignment document, citing each source with page numbers.

---

## Example 2: Cross-Document Comparison

**Question:** "Compare the technical skills mentioned in the resume vs the job description."

**Tool Called:** `compare_documents`

```json
{
  "document_a": "priyam-agrawal-sde-resume.pdf",
  "document_b": "JD_Software Engineeer (1).pdf",
  "aspect": "technical skills"
}
```

**Response:**
```json
{
  "comparison": "Document A (Resume) highlights hands-on experience with Python, Java, distributed systems, REST APIs, and cloud platforms. Document B (Job Description) emphasizes expertise in data integration, AI/ML pipelines, and the ability to work across no-code, low-code, and pro-code interfaces. Both documents share a common thread around full-stack engineering and production-grade system design. The resume demonstrates relevant experience that aligns with the role's requirements, particularly in backend development and API design.",
  "documents": ["priyam-agrawal-sde-resume.pdf", "JD_Software Engineeer (1).pdf"],
  "aspect": "technical skills"
}
```

**Demonstrates:** Multi-document comparison — the system retrieves relevant chunks from two separate documents and synthesizes a structured comparison on a specific aspect.

---

## Example 3: Document Summary

**Question:** "Summarize the take-home assignment document."

**Tool Called:** `get_document_summary`

```json
{
  "document_name": "Nexla_Software Engineer_Assignment.pdf"
}
```

**Response:**
```json
{
  "summary": "The Nexla Software Engineer Take-Home Assignment requires candidates to build a working MCP (Model Context Protocol) server that exposes document intelligence capabilities over 4-5 provided PDF files. The server must support document ingestion, natural language Q&A with source attribution, and multi-document awareness. Candidates are evaluated on code quality and architecture (35%), MCP protocol understanding (25%), and their AI-assisted development workflow or 'vibe coding' setup (40%). The assignment is estimated at 3-4 hours and must be delivered as a public GitHub repository with a comprehensive README.",
  "document": "Nexla_Software Engineer_Assignment.pdf",
  "chunks_used": 12
}
```

**Demonstrates:** Single-document summarization — the system retrieves key chunks from a specific document and generates a concise overview highlighting the main topics.

---

## Example 4: Autonomous Agent (Web UI)

When using the Web UI at `http://localhost:8000`, the LangChain agent autonomously selects which MCP tool to invoke:

**User message:** "What documents do I have?"

**Agent action:** Automatically calls `list_documents` tool via MCP.

**Agent response:**
> You currently have 3 documents in the knowledge base:
> 1. **JD_Software Engineeer (1).pdf** — 2 pages, 8 chunks
> 2. **Nexla_Software Engineer_Assignment.pdf** — 4 pages, 15 chunks
> 3. **priyam-agrawal-sde-resume.pdf** — 1 page, 4 chunks

**Demonstrates:** The LangGraph ReAct agent autonomously decided that `list_documents` was the appropriate tool for this query, called it over MCP stdio, and formatted the JSON response into natural language.
