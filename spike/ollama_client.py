"""Minimal Ollama /api/chat client (stdlib only)."""

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
REQUEST_TIMEOUT_S = 3600


@dataclass(frozen=True)
class ChatSettings:
    model: str
    cpu_only: bool
    think: bool | None  # None = don't send the flag (for models without a thinking mode)
    num_ctx: int = 16384
    num_predict: int = 6144


@dataclass(frozen=True)
class ChatResult:
    text: str
    seconds: float


def chat(settings: ChatSettings, prompt: str, images: list[Path]) -> ChatResult:
    message: dict[str, object] = {"role": "user", "content": prompt}
    if images:
        message["images"] = [base64.b64encode(p.read_bytes()).decode("ascii") for p in images]

    options: dict[str, object] = {
        "temperature": 0,
        "num_ctx": settings.num_ctx,
        "num_predict": settings.num_predict,
    }
    if settings.cpu_only:
        options["num_gpu"] = 0

    payload: dict[str, object] = {
        "model": settings.model,
        "messages": [message],
        "stream": False,
        "options": options,
    }
    if settings.think is not None:
        payload["think"] = settings.think

    started = time.perf_counter()
    body = _post(payload)
    seconds = time.perf_counter() - started
    text = body["message"]["content"]
    if not text.strip():
        thinking = body["message"].get("thinking", "")
        raise RuntimeError(
            f"{settings.model} returned an empty answer after {seconds:.0f}s "
            f"(done_reason={body.get('done_reason')}, {len(thinking)} chars of hidden thinking). "
            "Use a non-thinking (instruct) model tag, e.g. qwen3-vl:8b-instruct."
        )
    return ChatResult(text=text, seconds=seconds)


def _post(payload: dict[str, object]) -> dict:
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama error {e.code} for model {payload['model']}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError("Can't reach Ollama at localhost:11434 - is the Ollama app running?") from e
