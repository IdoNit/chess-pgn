"""Read a page image as text with GLM-OCR through a local Ollama server."""

import base64
from pathlib import Path
from typing import Protocol

import httpx

OCR_PROMPT = "Text Recognition:"


class ReaderUnavailable(Exception):
    """The text reader can't be reached, the model is missing, or it returned nothing."""


class PageIncomplete(Exception):
    """The reader stopped before the end of the page (its output was cut off)."""


class Reader(Protocol):
    def read(self, image: Path) -> str: ...


class OllamaReader:
    # A full book page is a large image plus a long answer: the defaults (a few thousand tokens) cut it off.
    CONTEXT_TOKENS = 16384
    ANSWER_TOKENS = 8192

    def __init__(
        self,
        base_url: str = "http://ollama:11434",
        model: str = "glm-ocr",
        timeout_s: float = 300,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.transport = transport

    def read(self, image: Path) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": OCR_PROMPT, "images": [base64.b64encode(image.read_bytes()).decode("ascii")]}
            ],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self.CONTEXT_TOKENS, "num_predict": self.ANSWER_TOKENS},
        }
        try:
            with httpx.Client(transport=self.transport, timeout=self.timeout_s) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.HTTPError as e:
            raise ReaderUnavailable(f"Can't reach the text reader at {self.base_url}") from e
        if response.status_code == 404:
            raise ReaderUnavailable(f"Model {self.model} is not installed in Ollama")
        if response.status_code != 200:
            raise ReaderUnavailable(f"Ollama answered {response.status_code}: {response.text[:200]}")
        body = response.json()
        text = body.get("message", {}).get("content", "")
        if body.get("done_reason") == "length":
            raise PageIncomplete(f"{self.model} stopped before the end of {image.name}")
        if not text.strip():
            raise ReaderUnavailable(f"{self.model} returned an empty answer")
        return text
