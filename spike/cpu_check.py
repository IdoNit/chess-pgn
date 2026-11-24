"""How fast would the models be on a computer without a GPU? Runs them on the CPU only, with 4 threads.

Uses the Ollama app you already have; "num_gpu": 0 keeps every layer on the CPU. From the project folder:

    uv run python -m spike.cpu_check
"""

import base64
import time
from pathlib import Path

import httpx

OLLAMA = "http://localhost:11434"
PAGE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "books" / "capablanca-1921" / "pages" / "p176.jpg"
CPU_ONLY = {"num_gpu": 0, "num_thread": 4, "temperature": 0}


def ask(model: str, prompt: str, image: bytes, extra: dict) -> float:
    started = time.monotonic()
    response = httpx.post(f"{OLLAMA}/api/chat", timeout=1800, json={
        "model": model, "stream": False, "keep_alive": "10m",
        "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(image).decode()]}],
        "options": CPU_ONLY | extra,
    })
    response.raise_for_status()
    return time.monotonic() - started


def main() -> None:
    page = PAGE.read_bytes()
    other = PAGE.with_name("p177.jpg").read_bytes()  # a different image: Ollama would reuse the work on the same one
    for model, prompt, extra, label in [
        ("glm-ocr", "Text Recognition:", {"num_ctx": 16384, "num_predict": 8192}, "GLM-OCR, one whole page"),
        ("qwen3-vl:8b-instruct", "Answer A or B: is this a page of a chess book?", {"num_predict": 5}, "vision model 8b, one question"),
        ("qwen3-vl:4b-instruct", "Answer A or B: is this a page of a chess book?", {"num_predict": 5}, "vision model 4b, one question"),
    ]:
        first = ask(model, prompt, page, extra)  # includes loading the model
        second = ask(model, prompt, other, extra)
        print(f"{label:34} {second:6.0f}s   (first call, with loading: {first:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
