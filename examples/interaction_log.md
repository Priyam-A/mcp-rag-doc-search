# Example Interaction Log

> These are sample Q&A interactions demonstrating the MCP server's capabilities.
> Source attribution is shown for each answer.

---

## Example 1: Single Document Query

**Question:** "What is Nexla's approach to data integration?"

**Tool Called:** `query_documents`

```json
{
  "question": "What is Nexla's approach to data integration?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "...(will be populated after testing with actual PDFs)...",
  "sources": [
    {
      "document": "example.pdf",
      "page": 1,
      "section": "Introduction",
      "relevance_score": 0.89
    }
  ],
  "query": "What is Nexla's approach to data integration?"
}
```

---

## Example 2: Multi-Document Query

**Question:** "Compare the key methodologies discussed across the documents."

**Tool Called:** `compare_documents`

```json
{
  "document_a": "paper1.pdf",
  "document_b": "paper2.pdf",
  "aspect": "methodology"
}
```

**Response:**
```json
{
  "comparison": "...(will be populated after testing with actual PDFs)...",
  "documents": ["paper1.pdf", "paper2.pdf"],
  "aspect": "methodology"
}
```

---

## Example 3: Document Summary

**Question:** "Summarize paper1.pdf"

**Tool Called:** `get_document_summary`

```json
{
  "document_name": "paper1.pdf"
}
```

**Response:**
```json
{
  "summary": "...(will be populated after testing with actual PDFs)...",
  "document": "paper1.pdf",
  "chunks_used": 12
}
```

---

> **Note:** This file will be updated with real responses once the provided PDF documents are ingested and tested. The examples above show the expected format and tool usage patterns.
