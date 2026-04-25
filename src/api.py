import logging
import shutil
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import config
from src.ingestion import DocumentStore, IngestionPipeline
from src.rag_engine import RAGEngine
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agent import get_agent

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Initialize core components (for /api/documents routes)
store = DocumentStore()
pipeline = IngestionPipeline(store=store)
engine = RAGEngine(store=store)

# Global agent references
mcp_agent_executor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mcp_agent_executor
    logger.info("Initializing MCP Client for LangChain Agent...")
    
    # We use sys.executable to ensure we use the same Python interpreter
    python_exec = sys.executable
    
    # Start the MCP client context manager
    mcp_client = MultiServerMCPClient({
        "nexla": {
            "transport": "stdio",
            "command": python_exec,
            "args": ["-m", "src.server"]
        }
    })
    
    # Fetch the tools exposed by src/server.py over stdio
    tools = await mcp_client.get_tools()
    logger.info(f"Loaded {len(tools)} tools from MCP server.")
    
    # Build our agent using these tools
    mcp_agent_executor = get_agent(tools)
    
    yield  # Run the FastAPI app
        
    logger.info("MCP Client connection closed.")

app = FastAPI(title="Nexla Document API", lifespan=lifespan)

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
async def chat(req: ChatRequest):
    if req.session_id not in chat_history:
        chat_history[req.session_id] = []
        
    if not mcp_agent_executor:
        raise HTTPException(status_code=500, detail="Agent is still initializing. Please try again.")
        
    messages = []
    for msg in chat_history[req.session_id]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
            
    messages.append(HumanMessage(content=req.question))
    
    try:
        # Invoke agent asynchronously because MCP tools are async
        response = await mcp_agent_executor.ainvoke({"messages": messages})
        final_message = response["messages"][-1].content
        
        chat_history[req.session_id].append({"role": "user", "content": req.question})
        chat_history[req.session_id].append({"role": "assistant", "content": final_message, "sources": []})
        
        return {"answer": final_message, "sources": []}
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return {"error": str(e)}

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
