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
        "You are Nexla Intelligence, an autonomous AI agent connected to a document knowledge base via MCP. "
        "You have the ability to execute tools to query, list, summarize, and compare PDF documents. "
        "1. When a user asks a question, YOU MUST ACTUALLY CALL the appropriate tool. "
        "2. DO NOT tell the user to use a tool. YOU must invoke the tool yourself to get the answer. "
        "3. YOU MUST ALWAYS cite the document name and page number directly in your final response. "
        "4. IMPORTANT: When calling tools, provide ONLY the raw literal string or integer values. DO NOT wrap strings in dictionaries. For example, pass `\"hire Priyam for Nexla\"` instead of `{\"type\": \"string\", \"value\": \"hire Priyam for Nexla\"}`."
        "5. If you do not know the answer, YOU MUST EXECUTE the `query_documents` tool to find it."
    )
        
    # Create the React agent
    agent_executor = create_react_agent(llm, tools, prompt=system_message)
    
    return agent_executor
