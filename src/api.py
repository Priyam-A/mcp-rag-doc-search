import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import config
from src.ingestion import DocumentStore, IngestionPipeline
from src.rag_engine import RAGEngine

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Initialize core components
store = DocumentStore()
pipeline = IngestionPipeline(store=store)
engine = RAGEngine(store=store)

app = FastAPI(title="Nexla Document API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory history: session_id -> list of messages
chat_history = {}

class ChatRequest(BaseModel):
    session_id: str = "default"
    question: str

@app.get("/api/documents")
def list_documents():
    docs = store.list_documents()
    return {"documents": docs, "total": len(docs)}

@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    config.documents_dir.mkdir(parents=True, exist_ok=True)
    file_path = config.documents_dir / file.filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        chunks = pipeline.ingest_pdf(file_path)
        return {"status": "success", "document": file.filename, "chunks": chunks}
    except Exception as e:
        logger.error(f"Failed to ingest {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{document_name}")
def delete_document(document_name: str):
    count = store.delete_document(document_name)
    file_path = config.documents_dir / document_name
    if file_path.exists():
        file_path.unlink()
    return {"status": "success", "document": document_name, "deleted_chunks": count}

@app.post("/api/chat")
def chat(req: ChatRequest):
    if req.session_id not in chat_history:
        chat_history[req.session_id] = []
        
    result = engine.query_documents(req.question, top_k=5)
    
    chat_history[req.session_id].append({"role": "user", "content": req.question})
    chat_history[req.session_id].append({"role": "assistant", "content": result["answer"], "sources": result.get("sources", [])})
    
    return result

@app.get("/api/history/{session_id}")
def get_history(session_id: str):
    return chat_history.get(session_id, [])

@app.delete("/api/history/{session_id}")
def clear_history(session_id: str):
    if session_id in chat_history:
        del chat_history[session_id]
    return {"status": "cleared"}

# Serve static frontend files
public_dir = Path(__file__).parent.parent / "public"
public_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(public_dir), html=True), name="public")
