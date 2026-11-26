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


HEADING_PROMPT = """These lines are the heading of a game in a chess book:

{heading}

Copy from it, exactly as printed:
- white: the player of the White pieces; black: the player of the Black pieces
- event: the tournament or match (not the game's title or number, and not the opening)
- site: the city or place where it was played
- year: the year it was played
- opening: the name of the opening, if the heading gives one
Join a word that is broken across two lines with a hyphen (for example "Dus-" then "Chotimirski" is one name,
"Dus-Chotimirski"). Leave a field empty ("") if the heading does not say. Answer in JSON:
{{"white": "", "black": "", "event": "", "site": "", "year": "", "opening": ""}}"""


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


    def read_headers(self, heading: str) -> dict[str, str] | None:
        """Players, event, site and year as the game's heading prints them (a reading question)."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": HEADING_PROMPT.format(heading=heading)}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 200, "num_ctx": 4096},
        }
        try:
            response = self.client.post(f"{self.base_url}/api/chat", json=payload)
            answer = json.loads(response.json()["message"]["content"]) if response.status_code == 200 else None
            return {k: str(v).strip() for k, v in answer.items()} if isinstance(answer, dict) else None
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return None


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
    if not _moves_read(white, black) or not _close(_NUMBERED.sub("", line, count=1), white, black):
        return None  # not moves, or not what the OCR saw: an invention, not a reading
    marked = " ".join(f"{m}{AI_MARK}" for m in (white, black) if m)
    return f"{number}. ... {marked}" if black_only else f"{number}. {marked}"


def _moves_read(white: str, black: str) -> bool:
    model = " ".join(m for m in (white, black) if m)
    return bool(model) and all(_readable(m) for m in _moves(model))


def _close(seen: str, white: str, black: str) -> bool:
    """Whether the model's reading is what another reader (the OCR, or tesseract) saw on that row."""
    ocr = _squash(seen)
    model = _squash(" ".join(m for m in (white, black) if m))
    return SequenceMatcher(a=ocr, b=model).ratio() >= CLOSE_ENOUGH and all(
        _kept(ocr, _squash(m)) >= CLOSE_ENOUGH for m in (white, black) if m)


def dropped_line(number: int, reading: tuple[str, str], tesseract_saw: str | None) -> str | None:
    """A row the OCR dropped altogether, as the model read it ("N. white† black†"), or None. When tesseract read
    that row too, the reading must be close to what tesseract saw."""
    white, black = reading
    if not white or not _moves_read(white, black):
        return None
    if tesseract_saw is not None and not _close(_NUMBERED.sub("", tesseract_saw, count=1), white, black):
        return None
    return f"{number}. " + " ".join(f"{m}{AI_MARK}" for m in (white, black) if m)


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
                 progress: Callable[[str], None] | None = None, flagged: set[int] = frozenset()) -> tuple[list[str], int]:
    """Ask about each damaged row numbered in `numbers`, and about each row numbered in `numbers` or `flagged`
    that the OCR lost altogether; return the page texts with accepted readings, and how many rows changed."""
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
    for number in sorted(_dropped(out, set(numbers) | set(flagged))):
        p, i, after = _neighbour(out, number)
        black_printed = _black_only_row(out, number)
        seen, box, q = None, None, p
        for q in (p, p - 1, p + 1):  # the lost row can sit at the foot of the previous page, or the next one's head
            if 0 <= q < len(images):
                if q not in pages:
                    image = Image.open(images[q])
                    pages[q] = (image, page_lines(image))
                seen, box = _tesseract_row(number, pages[q][1])
                if box is not None:
                    break
        if box is None:
            q = p
            box = _beside(out[p][i], number, after, pages[p][1])
        if box is None:
            continue
        strip = row_strip(pages[q][0], box)
        strip.info["dropped"] = True  # a row the OCR lost (the model's answer cannot be checked against the OCR)
        reading = row_reader.read_row(strip, number)
        if reading and black_printed:
            reading = (reading[0], "")  # Black's move is in the text already: only White's was lost
        new = dropped_line(number, reading, seen) if reading else None
        if new is not None:
            if black_printed:
                p, i = black_printed
                out[p].insert(i, new)
            else:
                out[p].insert(i + 1 if after else i, new)
            rewritten += 1
    for p, i, number in _white_only_rows(out, set(numbers) | set(flagged)):
        if p not in pages:
            image = Image.open(images[p])
            pages[p] = (image, page_lines(image))
        page, found = pages[p]
        box = locate(out[p][i], number, found)
        if box is None:
            continue
        strip = row_strip(page, box)
        strip.info["dropped"] = True  # Black's half of the row was lost
        reading = row_reader.read_row(strip, number)
        new = completed_line(out[p][i], reading) if reading else None
        if new is not None:
            out[p][i] = new
            rewritten += 1
    return ["\n".join(lines) for lines in out], rewritten


def _white_only_rows(pages: list[list[str]], numbers: set[int]) -> list[tuple[int, int, int]]:
    """(page, line, number) of rows numbered in `numbers` that hold White's move only, Black's being printed
    nowhere (no "N. ..." row): a table whose Black column the OCR lost."""
    black_rows = {n for lines in pages for line in lines if (m := _NUMBERED.match(line)) and m["dots"]
                  for n in [int(m.group(1))]}
    found = []
    for p, lines in enumerate(pages):
        for i, line in enumerate(lines):
            m = _NUMBERED.match(line)
            if not m or m["dots"] or int(m.group(1)) not in numbers or int(m.group(1)) in black_rows:
                continue
            moves = [t for t in _moves(_NUMBERED.sub("", line, count=1)) if not _WORD.match(t)]
            if len(moves) == 1 and _readable(moves[0]):
                found.append((p, i, int(m.group(1))))
    return found


def completed_line(line: str, reading: tuple[str, str]) -> str | None:
    """The row with Black's move added from the model's reading ("N. white black†"), or None. The model's White
    move must be the one the OCR read: that is how we know it read this row."""
    white, black = reading
    if not black or not _moves_read(white, black) or not _close(_NUMBERED.sub("", line, count=1), white, ""):
        return None
    return f"{line.rstrip()} {black}{AI_MARK}"


def _numbers(lines: list[str], white: bool = False) -> list[int | None]:
    """The move number each line starts with; with `white`, only rows that hold White's move ("N. ..." does not)."""
    return [int(m.group(1)) if (m := _NUMBERED.match(line)) and not (white and m["dots"]) else None for line in lines]


def _dropped(pages: list[list[str]], numbers: set[int]) -> set[int]:
    """Numbers asked about whose row with White's move is in none of the OCR's text, though the row itself (as
    "N. ... black"), or a neighbouring row, is."""
    whites = {n for lines in pages for n in _numbers(lines, white=True) if n is not None}
    present = {n for lines in pages for n in _numbers(lines) if n is not None}
    return {n for n in numbers if n not in whites and {n - 1, n, n + 1} & present}


def _black_only_row(pages: list[list[str]], number: int) -> tuple[int, int] | None:
    for p, lines in enumerate(pages):
        for i, line in enumerate(lines):
            if (m := _NUMBERED.match(line)) and int(m.group(1)) == number and m["dots"]:
                return p, i
    return None


def _neighbour(pages: list[list[str]], number: int) -> tuple[int, int, bool]:
    """(page, line, after): the row before the dropped one (after=True), else the row after it."""
    for wanted, after in ((number - 1, True), (number + 1, False), (number, False)):
        for p, lines in enumerate(pages):
            for i, n in enumerate(_numbers(lines)):
                if n == wanted:
                    return p, i, after
    raise LookupError(number)


def _tesseract_row(number: int, found: list[tuple[str, Box]]) -> tuple[str | None, Box | None]:
    """Tesseract's own line with White's move of that number, if it read one: (its text, its box)."""
    for text, box in found:
        if (m := _NUMBERED.match(text)) and int(m.group(1)) == number and not m["dots"]:
            return text, box
    return None, None


def _beside(neighbour: str, number: int, after: bool, found: list[tuple[str, Box]]) -> Box | None:
    """The space just below (or above) the neighbouring row, where the dropped row should be."""
    near = locate(neighbour, number - 1 if after else number + 1, found)
    if near is None:
        return None
    left, top, right, bottom = near
    height = bottom - top
    return (left, bottom, right, bottom + height) if after else (left, top - height, right, top)