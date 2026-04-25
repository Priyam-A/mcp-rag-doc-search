"""
LLM Provider abstraction for the Nexla MCP RAG Server.

Supports Ollama (local, free) and OpenAI (cloud, requires API key).
The provider is selected via the LLM_PROVIDER environment variable.
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "", history: list[dict] | None = None) -> str:
        """Generate a response from the LLM."""
        ...


class OllamaProvider(LLMProvider):
    """Local LLM via Ollama. No API key required."""

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        logger.info(f"Initializing Ollama provider with model: {model}")

    def generate(self, prompt: str, system_prompt: str = "", history: list[dict] | None = None) -> str:
        """Generate a response using Ollama."""
        try:
            import ollama

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": prompt})

            response = ollama.chat(model=self.model, messages=messages)
            return response["message"]["content"]

        except ImportError:
            raise RuntimeError(
                "Ollama package not installed. Run: pip install ollama"
            )
        except Exception as e:
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                raise RuntimeError(
                    f"Cannot connect to Ollama. Ensure Ollama is running: ollama serve\n"
                    f"Then pull the model: ollama pull {self.model}\n"
                    f"Original error: {e}"
                )
            raise RuntimeError(f"Ollama error: {e}")


class OpenAIProvider(LLMProvider):
    """Cloud LLM via OpenAI API."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.model = model
        logger.info(f"Initializing OpenAI provider with model: {model}")
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise RuntimeError(
                "OpenAI package not installed. Run: pip install openai"
            )

    def generate(self, prompt: str, system_prompt: str = "", history: list[dict] | None = None) -> str:
        """Generate a response using OpenAI."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,  # Low temp for factual, grounded answers
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"OpenAI error: {e}")


def create_llm_provider(
    provider: str, model: str = "", api_key: str = ""
) -> LLMProvider:
    """Factory function to create the appropriate LLM provider.

    Args:
        provider: Either "ollama" or "openai".
        model: Model name to use.
        api_key: API key (required for OpenAI).

    Returns:
        An initialized LLMProvider instance.
    """
    if provider == "openai":
        if not api_key:
            raise ValueError("OpenAI API key is required when using OpenAI provider.")
        return OpenAIProvider(api_key=api_key, model=model or "gpt-4o")
    else:
        return OllamaProvider(model=model or "llama3.2")
