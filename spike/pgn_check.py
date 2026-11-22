"""Legality checking and scoring of PGN text with python-chess."""

import io
import logging
import re
from dataclasses import dataclass

import chess
import chess.pgn

_FEN_IN_ERROR = re.compile(r" in ([1-8pnbrqkPNBRQK/]+ [wb] \S+ \S+ \d+ \d+)")
_FENCE = re.compile(r"```[a-zA-Z]*\n?")

# python-chess logs every parse error itself; we collect them from game.errors instead.
logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)


@dataclass(frozen=True)
class MoveProblem:
    message: str
    fen: str | None
    legal_moves: list[str]

    def describe(self) -> str:
        if self.fen is None:
            return self.message
        board = chess.Board(self.fen)
        side = "White" if board.turn == chess.WHITE else "Black"
        return (
            f"{self.message.split(' in ')[0]}\n"
            f"Position before that move (FEN): {self.fen}\n"
            f"It is {side} to move (move {board.fullmove_number}).\n"
            f"Legal moves here: {' '.join(self.legal_moves)}"
        )


@dataclass(frozen=True)
class Score:
    parsed: bool
    problems: int
    main_moves: int
    expected_main_moves: int
    main_prefix_ok: int
    main_exact: bool
    variation_moves: int
    expected_variation_moves: int
    variation_hits: int
    result_ok: bool


def clean_model_output(text: str) -> str:
    """Strip code fences / chatter before the PGN."""
    text = _FENCE.sub("", text).strip()
    for marker in ("[Event", "[", "1."):
        index = text.find(marker)
        if index != -1:
            return text[index:].strip() + "\n"
    return text + "\n"


def read(pgn_text: str) -> chess.pgn.Game | None:
    return chess.pgn.read_game(io.StringIO(pgn_text))


def problems(pgn_text: str) -> list[MoveProblem]:
    game = read(pgn_text)
    if game is None:
        return [MoveProblem("No game could be read from the output.", None, [])]
    found = [_to_problem(str(err)) for err in game.errors]
    if not found and game.next() is None:
        found.append(MoveProblem("The game has no moves.", None, []))
    return found


def _to_problem(message: str) -> MoveProblem:
    match = _FEN_IN_ERROR.search(message)
    if match is None:
        return MoveProblem(message, None, [])
    fen = match.group(1)
    board = chess.Board(fen)
    return MoveProblem(message, fen, sorted(board.san(m) for m in board.legal_moves))


def score(candidate_text: str, expected_text: str) -> Score:
    expected = read(expected_text)
    if expected is None:
        raise ValueError("Expected PGN could not be parsed")
    exp_main = [m.uci() for m in expected.mainline_moves()]
    exp_vars = _variation_nodes(expected)

    candidate = read(candidate_text)
    if candidate is None:
        return Score(False, 1, 0, len(exp_main), 0, False, 0, len(exp_vars), 0, False)

    cand_main = [m.uci() for m in candidate.mainline_moves()]
    cand_vars = _variation_nodes(candidate)
    return Score(
        parsed=True,
        problems=len(problems(candidate_text)),
        main_moves=len(cand_main),
        expected_main_moves=len(exp_main),
        main_prefix_ok=_common_prefix(cand_main, exp_main),
        main_exact=cand_main == exp_main,
        variation_moves=len(cand_vars),
        expected_variation_moves=len(exp_vars),
        variation_hits=len(cand_vars & exp_vars),
        result_ok=candidate.headers.get("Result") == expected.headers.get("Result"),
    )


def _variation_nodes(game: chess.pgn.Game) -> set[tuple[str, str]]:
    """(position before move, move) for every move that is not on the main line."""
    nodes: set[tuple[str, str]] = set()
    stack: list[tuple[chess.pgn.GameNode, bool]] = [(game, True)]
    while stack:
        node, on_main = stack.pop()
        for index, child in enumerate(node.variations):
            child_main = on_main and index == 0
            if not child_main:
                nodes.add((node.board().epd(), child.move.uci()))
            stack.append((child, child_main))
    return nodes


def _common_prefix(a: list[str], b: list[str]) -> int:
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count
