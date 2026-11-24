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


class _Looped(Exception):
    """Ollama aborted the answer because the model kept repeating the same tokens."""


class Reader(Protocol):
    def read(self, image: Path) -> str: ...


class OllamaReader:
    # A full book page is a large image plus a long answer: the defaults (a few thousand tokens) cut it off.
    CONTEXT_TOKENS = 16384
    ANSWER_TOKENS = 8192
    # Dotted leaders ("23. . . . . .") and diagrams can make the model repeat itself until Ollama aborts it.
    LOOP_RETRY_PENALTY = 1.15

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
        try:
            return self._read(image, {})
        except _Looped:
            pass
        try:
            return self._read(image, {"repeat_penalty": self.LOOP_RETRY_PENALTY})
        except _Looped as e:
            raise PageIncomplete(f"{self.model} kept repeating itself on {image.name}") from e

    def _read(self, image: Path, extra_options: dict[str, float]) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": OCR_PROMPT, "images": [base64.b64encode(image.read_bytes()).decode("ascii")]}
            ],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self.CONTEXT_TOKENS, "num_predict": self.ANSWER_TOKENS}
            | extra_options,
        }
        try:
            with httpx.Client(transport=self.transport, timeout=self.timeout_s) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.HTTPError as e:
            raise ReaderUnavailable(f"Can't reach the text reader at {self.base_url}") from e
        if response.status_code == 404:
            raise ReaderUnavailable(f"Model {self.model} is not installed in Ollama")
        if response.status_code == 500 and "repeat limit" in response.text:
            raise _Looped(response.text[:200])
        if response.status_code != 200:
            raise ReaderUnavailable(f"Ollama answered {response.status_code}: {response.text[:200]}")
        body = response.json()
        text = body.get("message", {}).get("content", "")
        if body.get("done_reason") == "length":
            raise PageIncomplete(f"{self.model} stopped before the end of {image.name}")
        if not text.strip():
            raise ReaderUnavailable(f"{self.model} returned an empty answer")
        return text
