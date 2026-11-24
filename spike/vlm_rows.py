"""EXPERIMENT: ask a small vision model two narrow questions about every numbered line of the book pages.

    Q1 "move row or paragraph?"  a close-up with the line boxed in red: a row of the move table, or a paragraph?
    Q2 "read this row"  a crop of just that line (and the line under it, for moves that wrap): copy its moves.

Answers are saved beside each page as pNNN.vlm8.json, so the pipeline can be tried offline afterwards.
Needs the Ollama app with the model pulled, and tesseract (to find where each line is on the page).
From the project folder:

    ollama pull qwen3-vl:8b-instruct
    uv run python -m spike.vlm_rows --try-one          # one page first: check the answers make sense
    uv run python -m spike.vlm_rows                    # all pages not done yet (safe to stop and re-run)
    uv run python -m spike.vlm_rows --model qwen3-vl:8b-instruct --redo
"""

import argparse
import base64
import io
import json
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import httpx
import pytesseract
from PIL import Image, ImageDraw

BOOKS = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "books"
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# A numbered line: "18. Kt (B3)x", "14. ... P-B4", "15) .... Q-KKt3; 16) ...", "21 N-K7ch K-B1".
_NUMBERED = re.compile(r"^\s*\$?\s*(\d{1,3})\s*[.)]?\s*(?:[.…]\s*)*\S")

Q1 = """This is a page from a chess book. Look at how the text inside the red box is laid out on the page.
(A) It is a row of a move table: a move number with one or two moves, set apart on its own line, like the
    other rows of the game's move list.
(B) It is part of a paragraph of running text (a note), even if that text contains moves.
Answer with one letter only: A or B."""

Q2 = """This image is a strip from a chess book. Find the row that starts with the move number {number}.
Copy the chess moves of that row exactly as printed, letter for letter, nothing else.
Answer in JSON: {{"white": "...", "black": "..."}}. Use "" for a side that has no move in that row
(for example when the row starts with dots, it gives only Black's move)."""


def numbered_lines(ocr_text: str) -> list[tuple[int, str]]:
    out = []
    for line in ocr_text.splitlines():
        m = _NUMBERED.match(line)
        if m and int(m.group(1)) <= 150:
            out.append((int(m.group(1)), line.strip()))
    return out


def page_lines(image: Image.Image) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Lines of text tesseract finds on the page, with their boxes (left, top, right, bottom)."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    lines: dict[tuple[int, int, int], list[int]] = {}
    for i, word in enumerate(data["text"]):
        if word.strip():
            lines.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]), []).append(i)
    out = []
    for ids in lines.values():
        text = " ".join(data["text"][i] for i in ids)
        box = (min(data["left"][i] for i in ids), min(data["top"][i] for i in ids),
               max(data["left"][i] + data["width"][i] for i in ids), max(data["top"][i] + data["height"][i] for i in ids))
        out.append((text, box))
    return out


def _squash(text: str) -> str:
    return re.sub(r"[\s$\\.…]|times", "", text).replace("—", "-").replace("×", "x").lower()


def locate(line: str, number: int, candidates: list[tuple[str, tuple[int, int, int, int]]],
           used: set) -> tuple[int, int, int, int] | None:
    """The box of `line` on the page: the best-matching tesseract line not given to another line yet (a paragraph
    matches by its first line); failing that, the first unused line that starts with the move number."""
    candidates = [(text, box) for text, box in candidates if box not in used]
    target = _squash(line)

    def similarity(text: str) -> float:
        squashed = _squash(text)
        return max(SequenceMatcher(a=target, b=squashed).ratio(),
                   SequenceMatcher(a=target[: len(squashed)], b=squashed).ratio() if len(target) > len(squashed) else 0)

    scored = [(similarity(text), box) for text, box in candidates]
    best = max(scored, default=(0.0, None))
    if best[0] >= 0.5:
        return best[1]
    starting = [(score, box) for (score, box), (text, _b) in zip(scored, candidates)
                if _squash(text).startswith(str(number))]
    return min(starting, key=lambda sb: sb[1][1])[1] if starting else None


def _png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def ask(model: str, prompt: str, image: Image.Image, as_json: bool) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [_png(image)]}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 200, "num_ctx": 8192},
    }
    if as_json:
        payload["format"] = "json"
    started = time.monotonic()
    response = httpx.post(f"{OLLAMA}/api/chat", json=payload, timeout=300)
    response.raise_for_status()
    return response.json()["message"]["content"].strip(), time.monotonic() - started


def boxed(page: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    marked = page.convert("RGB")
    draw = ImageDraw.Draw(marked)
    left, top, right, bottom = box
    draw.rectangle((left - 6, top - 5, right + 6, bottom + 5), outline=(230, 0, 0), width=3)
    return marked


def neighbourhood(page: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """The boxed line with a few lines above and below it, at full resolution (the model sees the layout)."""
    left, top, right, bottom = box
    height = bottom - top
    marked = boxed(page, box)
    return marked.crop((max(0, left - 250), max(0, top - 6 * height), min(page.width, right + 250),
                        min(page.height, bottom + 6 * height)))


def strip(page: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = box
    height = bottom - top
    return page.crop((max(0, left - 20), max(0, top - height // 2), min(page.width, right + 20),
                      min(page.height, bottom + int(height * 1.8))))  # the line under it, for wrapped moves


def run_page(image_path: Path, model: str) -> list[dict]:
    page = Image.open(image_path)
    found = page_lines(page)
    answers = []
    used: set = set()
    for number, line in numbered_lines(image_path.with_suffix(".ocr.txt").read_text(encoding="utf-8")):
        box = locate(line, number, found, used)
        if box is not None:
            used.add(box)
        entry: dict = {"number": number, "ocr": line, "box": box}
        if box is None:
            entry["error"] = "line not found on the page"
            answers.append(entry)
            continue
        kind, seconds_1 = ask(model, Q1, neighbourhood(page, box), as_json=False)
        reading, seconds_2 = ask(model, Q2.format(number=number), strip(page, box), as_json=True)
        entry |= {"kind": kind, "reading": reading, "seconds": round(seconds_1 + seconds_2, 1)}
        answers.append(entry)
    return answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-vl:8b-instruct")
    parser.add_argument("--suffix", default=".vlm8.json", help="answers are saved as pNNN<suffix>")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--try-one", action="store_true", help="ask about one page and print the answers")
    parser.add_argument("--page", default="capablanca-1921/p258.jpg", help="the page for --try-one")
    args = parser.parse_args()
    if args.try_one:
        for entry in run_page(BOOKS / args.page.replace("/", "/pages/"), args.model):
            print(entry.get("seconds"), "s |", entry["ocr"][:40], "|", entry.get("kind"), "|", entry.get("reading"))
        return
    pages = sorted(BOOKS.glob("*/pages/*.jpg"))
    todo = [p for p in pages if args.redo or not p.with_suffix(args.suffix).exists()]
    print(f"{len(todo)} of {len(pages)} pages to ask about, model {args.model}")
    started = time.monotonic()
    for n, page in enumerate(todo, start=1):
        answers = run_page(page, args.model)
        page.with_suffix(args.suffix).write_text(json.dumps({"model": args.model, "lines": answers}, indent=1),
                                                 encoding="utf-8")
        asked = [a for a in answers if "seconds" in a]
        left = (time.monotonic() - started) / n * (len(todo) - n)
        print(f"[{n}/{len(todo)}] {page.parent.parent.name}/{page.name}: {len(asked)} lines, "
              f"{sum(a['seconds'] for a in asked):.0f}s (about {left / 60:.0f} min left)")


if __name__ == "__main__":
    main()
