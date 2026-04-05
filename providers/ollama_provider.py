import requests
from providers.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    def __init__(self, model: str = "llama3"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate_response(self, prompt: str) -> str:
        response = requests.post(self.url, json={
            "model": self.model,
            "prompt": prompt,
            "stream": False
        })
        return response.json()["response"]