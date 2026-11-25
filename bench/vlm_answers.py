"""EXPERIMENT: feed the vision model's saved answers (spike.vlm_rows) into the pipeline's input text.

- A numbered line the model calls part of a paragraph (B) is wrapped in { } as commentary.
- A move row (A) whose OCR text doesn't read as moves is replaced by the model's reading of it, but only when
  that reading is close to what the OCR saw (it may regroup the OCR's characters, not invent new ones). Moves
  from such a row are flagged "read by AI" so a person glances at them.
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from chesspgn import notation
from chesspgn.tokens import normalize_notation, tokenize

CLOSE_ENOUGH = 0.6


def _squash(text: str) -> str:
    return re.sub(r"[\s.…$\\]", "", normalize_notation(text)).lower()


def _moves(text: str) -> list[str]:
    return [t.text for t in tokenize(normalize_notation(text)) if t.text and t.number is None]


def _readable(token: str) -> bool:
    return any(notation.read_clues(token, c) is not None for c in notation.CONVENTIONS)


class VlmAssisted:
    """A reader: the saved OCR text, adjusted with the model's answers. `repaired` collects (number, black?)
    of the moves taken from the model, so they can be flagged."""

    def __init__(self, suffix: str = ".vlm8.json", labels: bool = True, repair: bool = True,
                 tiebreak: bool = False) -> None:
        self.suffix, self.labels, self.repair, self.tiebreak = suffix, labels, repair, tiebreak
        self.repaired: set[tuple[int, bool]] = set()  # moves taken from the model, or lines it doubted: flagged
        self.last_row: tuple[int, bool] | None = None  # (number, ended with Black's move) of the last game row

    def reset(self) -> None:
        self.repaired, self.last_row = set(), None

    def _continues(self, number: int) -> bool:
        """Whether a row numbered `number` carries the game on from the last game row."""
        if self.last_row is None:
            return True
        last, ended_black = self.last_row
        return number in ((last + 1,) if ended_black else (last, last + 1))

    def _note_row(self, line: str, number: int) -> None:
        moves = _moves(line)
        black = bool(re.match(r"^\s*\d+\s*\.\s*(\.\s*){2,}", line)) or len(moves) >= 2
        self.last_row = (number, black)

    def read(self, image: Path) -> str:
        text = image.with_suffix(".ocr.txt").read_text(encoding="utf-8")
        answers = image.with_suffix(self.suffix)
        if not answers.exists():
            return text
        by_line = {a["ocr"]: a for a in json.loads(answers.read_text(encoding="utf-8"))["lines"] if "kind" in a}
        out = []
        for line in text.splitlines():
            answer = by_line.get(line.strip())
            paragraph = answer is not None and answer["kind"].strip().upper().startswith("B")
            if answer is None:
                out.append(line)
            elif self.labels and paragraph and (not self.tiebreak or not self._continues(answer["number"])):
                out.append("{" + line.replace("{", "(").replace("}", ")") + "}")
            elif self.tiebreak and paragraph:  # the model and the numbers disagree: keep it, and flag it
                self.repaired |= {(answer["number"], False), (answer["number"], True)}
                self._note_row(line, answer["number"])
                out.append(self._repaired(line, answer) if self.repair else line)
            else:
                self._note_row(line, answer["number"])
                out.append(self._repaired(line, answer) if self.repair else line)
        return "\n".join(out)

    def _repaired(self, line: str, answer: dict) -> str:
        ocr_moves = _moves(line)
        if ocr_moves and all(_readable(m) for m in ocr_moves):
            return line
        try:
            reading = json.loads(answer["reading"])
        except (json.JSONDecodeError, TypeError):
            return line
        white, black = str(reading.get("white", "")).strip(), str(reading.get("black", "")).strip()
        model = " ".join(m for m in (white, black) if m)
        if not model or not all(_readable(m) for m in _moves(model)):
            return line
        if SequenceMatcher(a=_squash(" ".join(ocr_moves)), b=_squash(model)).ratio() < CLOSE_ENOUGH:
            return line  # not what the OCR saw: an invention, not a reading
        number = answer["number"]
        black_only = not white or bool(re.match(r"^\s*\d+\s*\.\s*(\.\s*){2,}", line))
        if white and not black_only:
            self.repaired.add((number, False))
        if black:
            self.repaired.add((number, True))
        dots = " ..." if black_only else ""
        return f"{number}.{dots} {white if not black_only else ''} {black}".replace("  ", " ")


ROWS_SUFFIX = ".rows8.json"  # {"19": ["R-Q3", "P-QR4"]} or {"19": null} when the model could not say


class SavedRowReader:
    """A row reader that replays the vision model's saved answers (pNNN.vlm8.json), for the real repair path.

    The benchmark points `chesspgn.repair.row_strip` at `whole_page`, so the "strip" it receives is the page,
    which still knows its file name."""

    def __init__(self, suffix: str = ".vlm8.json") -> None:
        self.suffix = suffix

    def read_row(self, strip, number: int) -> tuple[str, str] | None:
        from chesspgn.repair import needs_reading

        if strip.info.get("dropped"):  # rows the OCR dropped: asked on their own crop, saved apart
            rows = Path(strip.filename).with_suffix(ROWS_SUFFIX)
            saved = json.loads(rows.read_text(encoding="utf-8")).get(str(number)) if rows.exists() else None
            return tuple(saved) if saved else None
        answers = Path(strip.filename).with_suffix(self.suffix)
        if not answers.exists():
            return None
        for entry in json.loads(answers.read_text(encoding="utf-8"))["lines"]:
            if entry["number"] == number and "reading" in entry and needs_reading(entry["ocr"]):
                try:
                    reading = json.loads(entry["reading"])
                except (json.JSONDecodeError, TypeError):
                    return None
                return str(reading.get("white", "")).strip(), str(reading.get("black", "")).strip()
        return None


def whole_page(page, box):
    copy = page.copy()  # a copy: the repair pass marks the "strip" it asks about
    copy.filename = page.filename
    return copy


from chesspgn.repair import row_strip as _app_row_strip  # noqa: E402 - kept before anyone replaces repair.row_strip


def named_strip(page, box):
    """The app's real crop of a row, still knowing its page's file name (for recording answers)."""
    strip = _app_row_strip(page, box)
    strip.filename = page.filename
    return strip
