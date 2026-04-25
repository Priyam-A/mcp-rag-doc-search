import logging
from langgraph.prebuilt import create_react_agent

from src.config import config

logger = logging.getLogger(__name__)

def get_agent(tools: list):
    """Factory to create a LangGraph agent wrapping dynamic MCP tools."""
    
    # Initialize the LLM based on config
    if config.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=config.openai_model, api_key=config.openai_api_key)
        logger.info("Agent initialized with ChatOpenAI")
    else:
        # Use ChatOllama for local
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=config.ollama_model)
        logger.info("Agent initialized with ChatOllama")
        
    system_message = (
        "You are Nexla Intelligence, a highly capable AI agent connected to a document knowledge base via MCP. "
        "You have access to tools that can query, list, summarize, and compare PDF documents. "
        "1. When a user asks a question, decide which tool is best to use. "
        "2. If you use `query_documents`, you will receive an answer along with sources. "
        "3. YOU MUST ALWAYS cite the document name and page number directly in your final response to the user. "
        "4. Be concise, professional, and helpful."
    )
        
    # Create the React agent
    agent_executor = create_react_agent(llm, tools, prompt=system_message)
    
    return agent_executor
