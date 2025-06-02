# ollama_llm.py

from typing import Optional, List, Mapping, Any
import requests

from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation


class Ollama(LLM):
    """
    A LangChain-compatible LLM wrapper around a local Ollama instance.
    Sends a request to Ollama's /api/generate (or /chat/completions) endpoint
    and returns the generated text.
    """

    model: str = "gemma3:1b"  # e.g. "gemma3:1b", or "llama3:2b", etc.
    base_url: str = "http://localhost:11434"
    temperature: float = 0.3
    max_tokens: int = 128000

    def __init__(
        self,
        model: str = "gemma3:1b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_tokens: int = 128000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def _llm_type(self) -> str:
        return "ollama"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Core method that LangChain calls under the hood when you do llm(prompt).
        We'll send a non-streaming request to /api/generate and return data['response'].
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,  # ensure single JSON response
        }

        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        if "response" not in data:
            raise ValueError(f"Unexpected Ollama reply format: {data}")

        return data["response"]

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """
        Batch is not fully supported for Ollama (since the HTTP API is single-prompt).
        We'll just loop over prompts and call _call() for each one.
        """
        generations = []
        for p in prompts:
            text = self._call(p, stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations, llm_output={})
