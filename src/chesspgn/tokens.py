"""Split book text into tokens, and list the OCR corrections a token could need."""

import re
from dataclasses import dataclass
from functools import lru_cache

from chesspgn.notation import GROUP_SYMBOLS

COST_SUBSTITUTION = 1.0  # one OCR-confusable character changed
COST_DELETION = 1.5  # one character the OCR invented
COST_INSERTION = 1.2  # one character the OCR lost
COST_DROP_GROUP = 0.8  # an icon group symbol attached to the wrong move

_INSERTABLE = "KQRBNP12345678abcdefghx"
_CONFUSIONS: list[tuple[str, str]] = [
    ("S", "5"), ("O", "0"), ("l", "1"), ("I", "1"), ("B", "8"), ("Z", "2"), ("g", "9"),
    ("b", "6"), ("e", "c"), ("h", "b"), ("K", "R"), ("N", "H"), ("B", "R"), ("Q", "O"),
    ("x", "X"), ("-", "x"), ("1", "7"), ("3", "8"), ("6", "5"), ("♗", "♕"), ("♖", "♔"), ("N", "M"),
]
_SUBSTITUTES: dict[str, set[str]] = {}
for _a, _b in _CONFUSIONS:
    _SUBSTITUTES.setdefault(_a, set()).add(_b)
    _SUBSTITUTES.setdefault(_b, set()).add(_a)

_DASHES = re.compile(r"[–—−‒]+|-{2,}")
_SPACED_DOTS = re.compile(r"\.\s+\.\s*\.")
_NUMBER = re.compile(r"^(?P<n>\d{1,3})(?P<dots>\.{1,3})?$")
# Rank "1" read as "l" at the end of a move ("Rgl", "Q×hl+", "Rael"): a routine OCR quirk, fixed silently.
_L_FOR_ONE = re.compile(r"^([KQRBN♔♕♖♗♘♚♛♜♝♞]?[a-h]?x?[a-h])l([+#]*)$")
_OCR_NUMBER = re.compile(r"^(?=.*\d|[lI]{2})[\dlI]{2,3}(?P<dots>\.{0,3})$")  # "ll" = 11, "l3" = 13
_NUMBER_GLUED = re.compile(r"^(?P<n>\d{1,3})(?P<dots>\.{1,3})(?P<rest>[^.\s]\S*)$")
_ANNOTATION = re.compile(r"[!?]+$")
RESULT = re.compile(r"^(1-0|0-1|1:0|0:1|½-½|1/2-1/2)$")


@dataclass(frozen=True)
class Token:
    text: str
    annotation: str = ""
    number: int | None = None
    black_to_move: bool = False


def tokenize(text: str) -> list[Token]:
    text = _DASHES.sub("-", text).replace("…", "...").replace("×", "x")  # "e×d5" is how many books print captures
    text = _SPACED_DOTS.sub("...", text)  # "16 . . . N-Q2"
    tokens: list[Token] = []
    for raw in text.split():
        tokens.extend(_split_word(raw.strip(",;")))
    return _attach_lone_dots(tokens)


def _split_word(raw: str) -> list[Token]:
    if not raw or raw in ("...", ".."):
        return [Token("...")] if raw else []
    out: list[Token] = []
    if _OCR_NUMBER.match(raw):
        raw = raw.replace("l", "1").replace("I", "1")
    glued = _NUMBER_GLUED.match(raw)
    if glued is not None and RESULT.match(raw) is None:
        out.append(Token("", number=int(glued["n"]), black_to_move=len(glued["dots"]) >= 2))
        raw = glued["rest"]
    number = _NUMBER.match(raw)
    if number is not None:
        out.append(Token("", number=int(number["n"]), black_to_move=len(number["dots"] or "") >= 2))
        return out
    annotation = _ANNOTATION.search(raw)
    body = raw[: annotation.start()] if annotation else raw
    body = _L_FOR_ONE.sub(r"\g<1>1\g<2>", body)
    if body:
        out.append(Token(body, annotation=annotation.group(0) if annotation else ""))
    return out


def _attach_lone_dots(tokens: list[Token]) -> list[Token]:
    """'16 ... N-Q2': the dots make the preceding number a Black move number."""
    out: list[Token] = []
    for token in tokens:
        if token.text == "...":
            if out and out[-1].number is not None:
                out[-1] = Token("", number=out[-1].number, black_to_move=True)
            continue
        out.append(token)
    return out


@lru_cache(maxsize=200_000)
def variants(text: str) -> tuple[tuple[str, float], ...]:
    """(variant, cost) pairs: the token itself at cost 0, then OCR corrections, cheapest first."""
    found: dict[str, float] = {text: 0.0}
    for variant, cost in _one_edit(text):
        found.setdefault(variant, cost)
    for variant, cost in list(found.items()):
        if 0 < cost <= COST_SUBSTITUTION:
            for second, extra in _one_edit(variant):
                found.setdefault(second, cost + extra)
    for variant, cost in _insertions(text):  # one layer only: keeps the variant count bounded
        found.setdefault(variant, cost)
    return tuple(sorted(found.items(), key=lambda kv: kv[1]))


def _one_edit(text: str):
    if text[:1] in GROUP_SYMBOLS:  # icon linked to the wrong move: dropping it is the cheapest explanation
        yield text[1:], COST_DROP_GROUP
    for i, ch in enumerate(text):
        for sub in _SUBSTITUTES.get(ch, ()):
            yield text[:i] + sub + text[i + 1:], COST_SUBSTITUTION
        if len(text) > 2:
            yield text[:i] + text[i + 1:], COST_DELETION
    if "x" not in text and len(text) >= 2:  # a dropped capture sign: "PP" -> "PxP"
        for i in range(1, len(text)):
            yield text[:i] + "x" + text[i:], COST_SUBSTITUTION


def _insertions(text: str):
    """A character the OCR lost entirely ("xN" -> "RxN", "KR-K" -> "KR-K1")."""
    for i in range(len(text) + 1):
        for ch in _INSERTABLE:
            yield text[:i] + ch + text[i:], COST_INSERTION
