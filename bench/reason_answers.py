"""Replay the reasoning model's saved answers (spike/reason_stretches.py) into a decoded game.

Each answer is checked move by move against the page text before anything of it is used:
    the printed text reads as the model's move (at most one OCR correction)         -> accepted
    nothing printed there, or the printed text reads as no legal move at all        -> accepted (no evidence against)
    the printed text reads clearly as another legal move                            -> rejected, with the rest
Accepted moves are played (as fixes the decoder must keep) and every one of them stays flagged "reasoned".
"""

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.pgn

from bench.extraction import numbered_entries
from chesspgn import notation
from chesspgn.decoder import _NO_MAPPING, Flag, _reading_match, _readings
from chesspgn.pipeline import ConvertedGame, game_from_text
from chesspgn.tokens import AI_MARK, COST_SUBSTITUTION


@dataclass
class Reasoned:
    game: ConvertedGame
    pins: dict[int, chess.Move] = field(default_factory=dict)
    weak: set[int] = field(default_factory=set)  # plies whose move the page does not clearly print: flagged
    asked: int = 0
    accepted: int = 0  # moves
    rejected: int = 0  # answers cut short by the page text
    invalid: int = 0  # answers with an illegal or unparsable move (the rest of that answer is dropped)


def _best_cost(board: chess.Board, move: chess.Move, printed: list[str]) -> float | None:
    costs = [found[0] for text in printed for convention in notation.CONVENTIONS
             if (found := _reading_match(board, move, _readings(text.replace(AI_MARK, "").strip("'\"`‘’“”"), convention), _NO_MAPPING))]
    return min(costs, default=None)


def evidence(board: chess.Board, move: chess.Move, printed: list[str]) -> str | None:
    """How the page supports `move`: "exact" (prints it), "weak" (one OCR correction, or nothing readable
    printed), or None (the page clearly prints another legal move)."""
    own = _best_cost(board, move, printed) if printed else None
    if own is not None and own < COST_SUBSTITUTION:
        return "exact"
    if own is not None and own <= COST_SUBSTITUTION:
        return "weak"
    others = [c for m in board.legal_moves if m != move and (c := _best_cost(board, m, printed)) is not None]
    return None if any(c <= COST_SUBSTITUTION for c in others) else "weak"


def fits_the_page(board: chess.Board, move: chess.Move, printed: list[str]) -> bool:
    return evidence(board, move, printed) is not None


def read_answer(board: chess.Board, text: str) -> chess.Move | None:
    """The model's move: SAN, or (if it answered in the book's own notation) the one legal move that reads exactly."""
    try:
        return board.parse_san(text)
    except ValueError:
        exact = [m for m in board.legal_moves if (c := _best_cost(board, m, [text])) is not None and c == 0]
        return exact[0] if len(exact) == 1 else None


def accepted(board: chess.Board, answer: list[str], entries: dict, count: int,
             weak: list[bool] | None = None) -> tuple[list[chess.Move], str]:
    """The moves of an answer that are legal and fit the page, up to the first that is not: (moves, why it stopped).
    `weak` (if given) gets, per accepted move, whether the page only weakly supports it."""
    board = board.copy()
    played: list[chess.Move] = []
    for san in answer[:count]:
        move = read_answer(board, san)
        if move is None:
            return played, "invalid"
        found = evidence(board, move, entries.get((board.fullmove_number, board.turn == chess.BLACK), []))
        if found is None:
            return played, "rejected"
        if weak is not None:
            weak.append(found == "weak")
        played.append(move)
        board.push(move)
    return played, "ok"


def board_before(game: ConvertedGame, ply: int) -> chess.Board:
    board = chess.Board()
    for move in list(chess.pgn.read_game(io.StringIO(game.pgn)).mainline_moves())[: ply - 1]:
        board.push(move)
    return board


def step(original: ConvertedGame, current: ConvertedGame, pins: dict[int, chess.Move], start: int, end: int,
         answer: list[str], entries: dict, weak_plies: set[int] | None = None) -> tuple[ConvertedGame, str, int]:
    """Use one answer: accepted moves join the pins and the game is decoded again with them (in place in `pins`)."""
    board = board_before(current, start)
    if board.ply() != start - 1:  # the game is shorter than where this stretch starts
        return current, "invalid", 0
    weak: list[bool] = []
    played, why = accepted(board, answer, entries, end - start + 1, weak)
    for offset, move in enumerate(played):
        pins[start + offset] = move
        if weak_plies is not None and weak[offset]:
            weak_plies.add(start + offset)
    if played:
        current = game_from_text(original.text, dict(pins), original.headers)
    return current, why, len(played)


def apply(game: ConvertedGame, answers_file: Path, model: str) -> Reasoned:
    result = Reasoned(game)
    if not answers_file.exists():
        return result
    stretches = [s for s in json.loads(answers_file.read_text(encoding="utf-8"))["stretches"]
                 if s.get("model") == model and not s["error"] and s.get("v", 0) >= 2]
    entries = numbered_entries(game.text)
    current = game
    for stretch in sorted(stretches, key=lambda s: s["start"]):
        result.asked += 1
        current, why, count = step(game, current, result.pins, stretch["start"], stretch["end"], stretch["moves"],
                                   entries, result.weak)
        result.accepted += count
        result.rejected += why == "rejected"
        result.invalid += why == "invalid"
    if result.pins:
        reasoned = [Flag(ply, "", "", "reasoned") for ply in sorted(result.weak)]
        current.flags = [f for f in current.flags if f.ply not in result.pins or f.reason.startswith("side:")] + reasoned
    result.game = current
    return result
