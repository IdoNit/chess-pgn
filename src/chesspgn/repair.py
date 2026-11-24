"""A second reading of damaged rows by a small vision model, used only where the board can't read the OCR's.

The model is asked one narrow question per row: "copy the moves of the row numbered N", shown a strip cropped
around that line. Its reading replaces the OCR's only when the OCR's doesn't read as moves and the model's is
close to it (it may regroup what the OCR saw, e.g. a wrapped "B" glued to the wrong move, but not invent moves).
Moves taken from it are written with a trailing "†", so the decoder flags them "read by AI".
"""

import base64
import io
import json
import re
from collections.abc import Callable
from difflib import SequenceMatcher
from pathlib import Path
from typing import Protocol

import httpx
import pytesseract
from PIL import Image

from chesspgn import notation
from chesspgn.tokens import AI_MARK, normalize_notation, tokenize

CLOSE_ENOUGH = 0.6  # share of the OCR's characters, in order, that the model's reading must keep
_NUMBERED = re.compile(r"^\s*\$?\s*(\d{1,3})\s*[.)]?\s*(?P<dots>(?:[.…]\s*){2,})?")
_WORD = re.compile(r"^[A-Za-z][a-z]+[.,;:!?]*$")

Box = tuple[int, int, int, int]


class RowReader(Protocol):
    def read_row(self, strip: Image.Image, number: int) -> tuple[str, str] | None:
        """White's and Black's move as printed in the row numbered `number`, or None if it can't say."""
        ...


ROW_PROMPT = """This image is a strip from a chess book. Find the row that starts with the move number {number}.
Copy the chess moves of that row exactly as printed, letter for letter, nothing else.
Answer in JSON: {{"white": "...", "black": "..."}}. Use "" for a side that has no move in that row
(for example when the row starts with dots, it gives only Black's move)."""


class OllamaRowReader:
    def __init__(self, base_url: str = "http://ollama:11434", model: str = "qwen3-vl:8b-instruct",
                 timeout_s: float = 600, transport: httpx.BaseTransport | None = None) -> None:
        self.base_url, self.model = base_url.rstrip("/"), model
        self.client = httpx.Client(timeout=timeout_s, transport=transport)

    def read_row(self, strip: Image.Image, number: int) -> tuple[str, str] | None:
        buffer = io.BytesIO()
        strip.save(buffer, format="PNG")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": ROW_PROMPT.format(number=number),
                          "images": [base64.b64encode(buffer.getvalue()).decode("ascii")]}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 200, "num_ctx": 8192},
        }
        try:
            response = self.client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.HTTPError:
            return None  # the helper is optional: without an answer, the row just stays flagged
        if response.status_code != 200:
            return None
        try:
            answer = json.loads(response.json()["message"]["content"])
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
        if not isinstance(answer, dict):
            return None
        return str(answer.get("white", "")).strip(), str(answer.get("black", "")).strip()


# ---------------------------------------------------------------- deciding whether to use a reading

def _moves(text: str) -> list[str]:
    return [t.text for t in tokenize(normalize_notation(text)) if t.text and t.number is None]


def _readable(move: str) -> bool:
    return any(notation.read_clues(move, convention) is not None for convention in notation.CONVENTIONS)


def _squash(text: str) -> str:
    """Characters only, spelled one way: compares what the OCR and the model saw, whatever their spacing."""
    plain = re.sub(r"[\s.…$\\()@]", "", text).lower()
    return re.sub(r"[—–−]", "-", plain).replace("×", "x").replace("kt", "n")


def _kept(ocr: str, move: str) -> float:
    """Share of the move's characters found, in order, in the OCR's row."""
    matched = sum(block.size for block in SequenceMatcher(a=ocr, b=move, autojunk=False).get_matching_blocks())
    return matched / len(move) if move else 1.0


def needs_reading(line: str) -> bool:
    """Whether the OCR's row fails to read as moves."""
    moves = [m for m in _moves(_NUMBERED.sub("", line, count=1)) if not _WORD.match(m)]
    return not moves or not all(_readable(m) for m in moves)


def repaired_line(line: str, number: int, reading: tuple[str, str]) -> str | None:
    """The row rewritten with the model's reading ("N. white† black†"), or None to keep the OCR's."""
    if not needs_reading(line):
        return None
    white, black = reading
    numbered = _NUMBERED.match(line)
    black_only = bool(numbered and numbered["dots"]) or not white
    white = "" if black_only else white
    model = " ".join(m for m in (white, black) if m)
    if not model or not all(_readable(m) for m in _moves(model)):
        return None
    ocr = _squash(_NUMBERED.sub("", line, count=1))
    if SequenceMatcher(a=ocr, b=_squash(model)).ratio() < CLOSE_ENOUGH or not all(
            _kept(ocr, _squash(m)) >= CLOSE_ENOUGH for m in (white, black) if m):
        return None  # not what the OCR saw: an invention, not a reading
    marked = " ".join(f"{m}{AI_MARK}" for m in (white, black) if m)
    return f"{number}. ... {marked}" if black_only else f"{number}. {marked}"


# ---------------------------------------------------------------- finding a row on the page

def page_lines(image: Image.Image) -> list[tuple[str, Box]]:
    """Lines of text tesseract finds on the page, with their boxes (left, top, right, bottom)."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    lines: dict[tuple[int, int, int], list[int]] = {}
    for i, word in enumerate(data["text"]):
        if word.strip():
            lines.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]), []).append(i)
    return [(" ".join(data["text"][i] for i in ids),
             (min(data["left"][i] for i in ids), min(data["top"][i] for i in ids),
              max(data["left"][i] + data["width"][i] for i in ids), max(data["top"][i] + data["height"][i] for i in ids)))
            for ids in lines.values()]


def _plain(text: str) -> str:
    return re.sub(r"[\s$\\.…]|times", "", text).replace("—", "-").replace("×", "x").lower()


def locate(line: str, number: int, candidates: list[tuple[str, Box]]) -> Box | None:
    """The box of `line`: the best-matching tesseract line, else the first one starting with the move number."""
    target = _plain(line)

    def similarity(text: str) -> float:
        plain = _plain(text)
        prefix = SequenceMatcher(a=target[: len(plain)], b=plain).ratio() if len(target) > len(plain) else 0
        return max(SequenceMatcher(a=target, b=plain).ratio(), prefix)

    scored = [(similarity(text), box) for text, box in candidates]
    best = max(scored, default=(0.0, None))
    if best[0] >= 0.5:
        return best[1]
    starting = [box for text, box in candidates if _plain(text).startswith(str(number))]
    return min(starting, key=lambda b: b[1]) if starting else None


def row_strip(page: Image.Image, box: Box) -> Image.Image:
    """The line, a little above it, and the line under it (moves that wrap continue there)."""
    left, top, right, bottom = box
    height = bottom - top
    return page.crop((max(0, left - 20), max(0, top - height // 2), min(page.width, right + 20),
                      min(page.height, bottom + int(height * 1.8))))


# ---------------------------------------------------------------- the pass over the pages

def repair_pages(images: list[Path], texts: list[str], numbers: set[int], row_reader: RowReader,
                 progress: Callable[[str], None] | None = None) -> tuple[list[str], int]:
    """Ask about each damaged row numbered in `numbers`; return the page texts with accepted readings, and how
    many rows were rewritten."""
    asks = [(p, i) for p, text in enumerate(texts) for i, line in enumerate(text.splitlines())
            if (m := _NUMBERED.match(line)) and int(m.group(1)) in numbers and needs_reading(line)]
    pages: dict[int, tuple[Image.Image, list[tuple[str, Box]]]] = {}
    out = [text.splitlines() for text in texts]
    rewritten = 0
    for asked, (p, i) in enumerate(asks, start=1):
        if progress:
            progress(f"{asked}/{len(asks)}")
        if p not in pages:
            image = Image.open(images[p])
            pages[p] = (image, page_lines(image))
        page, found = pages[p]
        line = out[p][i]
        number = int(_NUMBERED.match(line).group(1))
        box = locate(line, number, found)
        if box is None:
            continue
        reading = row_reader.read_row(row_strip(page, box), number)
        new = repaired_line(line, number, reading) if reading else None
        if new is not None:
            out[p][i] = new
            rewritten += 1
    return ["\n".join(lines) for lines in out], rewritten
