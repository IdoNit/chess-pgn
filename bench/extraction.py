"""Measure the reader on its own, apart from how the game is put together.

For every move of the book's game: is it in the OCR text, under its own move number and side, and how well
does it read there? Exact, one OCR correction away (fixable), present but unreadable (damaged), or absent.
"""

import re
from dataclasses import dataclass, field

import chess

from chesspgn import notation
from chesspgn.decoder import _NO_MAPPING, _reading_match, _readings
from chesspgn.tokens import COST_SUBSTITUTION, normalize_notation, tokenize

_WORD = re.compile(r"^[A-Za-z][a-z]+[.,;:!?]*$")


@dataclass
class Extraction:
    exact: int = 0
    fixable: int = 0
    damaged: int = 0
    missing: int = 0
    bad_plies: list[int] = field(default_factory=list)  # damaged or missing, 1-based


def numbered_entries(text: str) -> dict[tuple[int, bool], list[str]]:
    """Every token printed in a move slot: (move number, black?) -> texts, wherever it appears on the page."""
    clean = re.sub(r"[{}\[\]]", " ", normalize_notation(text))
    entries: dict[tuple[int, bool], list[str]] = {}
    slot: tuple[int, bool] | None = None
    for token in tokenize(clean):
        if token.number is not None:
            slot = (token.number, token.black_to_move)
            continue
        if slot is None or not token.text or _WORD.match(token.text):
            slot = None
            continue
        entries.setdefault(slot, []).append(token.text)
        slot = (slot[0], True) if not slot[1] else None
    return entries


def extraction(text: str, expected: list[str]) -> Extraction:
    entries = numbered_entries(text)
    result = Extraction()
    board = chess.Board()
    for ply, san in enumerate(expected, start=1):
        move = board.parse_san(san)
        printed = entries.get((board.fullmove_number, board.turn == chess.BLACK), [])
        costs = [found[0] for text in printed for convention in notation.CONVENTIONS
                 if (found := _reading_match(board, move, _readings(text, convention), _NO_MAPPING))]
        best = min(costs, default=None)
        if best is not None and best < COST_SUBSTITUTION:
            result.exact += 1
        elif best is not None and best <= COST_SUBSTITUTION:
            result.fixable += 1
        else:
            if printed:
                result.damaged += 1
            else:
                result.missing += 1
            result.bad_plies.append(ply)
        board.push(move)
    return result
