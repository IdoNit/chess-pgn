"""Tell a book's game text from its commentary by page layout, for books that don't bracket their comments.

Outside brackets, a line belongs to the commentary when it starts with an ordinary word, when it continues a
commentary paragraph (same paragraph: no blank line in between) without being a plain move line, or when it
is mostly words. Commentary lines are wrapped in { } so the decoder ignores them. Lines that end the game
("White resigns", "Resigns", "and won", "1-0") always stay game text. Bracketed text is left as it is.
"""

import re

from chesspgn import decoder
from chesspgn.tokens import Token

# How books end a game. A bare "wins"/"won" inside analysis ("... B×f4 wins") is not a result.
_RESULT = re.compile(
    r"\b(?:White|Black)\s+resign|^\s*Resigns?\b|\bresigns?\.?\s*$|\band\s+won\b|^\s*Drawn\b|(?<![\w-])(?:1-0|0-1|1:0|0:1|½-½)(?![\w-])",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"^(?:\d{1,3}|[lI\d]{2,3})\.{0,3}$|^\.{2,3}$")
_WORD = re.compile(r"^[^\W\d_][^\W\d_'’\-]+$")
_EDGE_PUNCTUATION = "\"'“”‘’,.;:!?()[]{}"
MIN_WORDS_FOR_PROSE = 3


def mark_commentary(text: str) -> str:
    depth = _depths(text)
    out: list[str] = []
    previous_was_commentary = False
    position = 0
    for line in text.splitlines(keepends=True):
        start, end = position, position + len(line.rstrip("\r\n"))
        position += len(line)
        outside = "".join(ch for i, ch in enumerate(text[start:end], start) if depth[i] == 0 and ch not in "()[]{}")
        tokens = outside.split()
        if not line.strip():
            previous_was_commentary = False
            out.append(line)
            continue
        if not tokens:
            out.append(line)  # entirely inside brackets: already a side line or a comment
            continue
        commentary = _RESULT.search(outside) is None and _is_commentary(tokens, previous_was_commentary)
        previous_was_commentary = commentary
        out.append(_wrap(text, start, end, depth) + line[end - start:] if commentary else line)
    return "".join(out)


def _is_commentary(tokens: list[str], continues_commentary: bool) -> bool:
    kinds = [_kind(t) for t in tokens]
    words, moves = kinds.count("word"), kinds.count("move")
    if kinds[0] == "word":
        return True
    if continues_commentary and (kinds[0] != "number" or words >= 1):
        return True
    return words >= MIN_WORDS_FOR_PROSE and words > moves


def _kind(token: str) -> str:
    bare = token.strip(_EDGE_PUNCTUATION)
    if _NUMBER.match(token) and any(ch.isdigit() or ch in "lI" for ch in token) or token in ("...", ".."):
        return "number"
    candidate = bare.replace("×", "x")
    if candidate and any(decoder.looks_like_a_move(Token(candidate), c) for c in ("english", "descriptive")):
        return "move"
    if _WORD.match(bare):
        return "word"
    return "other"


def _wrap(text: str, start: int, end: int, depth: list[int]) -> str:
    line = text[start:end]
    if depth[start] == 0 and (end == start or _depth_after(text, end - 1, depth) == 0):
        return "{" + line + "}"
    # The line starts or ends inside brackets: wrap only its outside parts, keeping brackets balanced.
    pieces, current = [], []
    for i in range(start, end):
        if depth[i] == 0 and text[i] not in "([{":
            current.append(text[i])
            continue
        if current:
            pieces.append("{" + "".join(current) + "}")
            current = []
        pieces.append(text[i])
    if current:
        pieces.append("{" + "".join(current) + "}")
    return "".join(pieces)


def _depths(text: str) -> list[int]:
    depth, out = 0, []
    for ch in text:
        if ch in "([{":
            depth += 1
            out.append(depth)
            continue
        out.append(depth)
        if ch in ")]}" and depth > 0:
            depth -= 1
    return out


def _depth_after(text: str, index: int, depth: list[int]) -> int:
    return depth[index] - (1 if text[index] in ")]}" else 0)
