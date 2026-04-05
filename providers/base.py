from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    Any new provider (Groq, Ollama, Anthropic, OpenAI)
    must implement the `complete` method.
    """
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Send a prompt, get a string response back."""
        pass
    