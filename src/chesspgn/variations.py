"""Build a full PGN game (main line + side lines) from one game's book text.

Structure comes from the text: ( ) / [ ] groups, with ';' separating alternatives. Placement comes from the
board: each alternative is tried at every line where its move number fits, and kept where its moves play.
"""

from dataclasses import dataclass, field

import time

import chess
import chess.pgn

from chesspgn import decoder
from chesspgn.decoder import Decoded, Flag
from chesspgn.layout import mark_commentary
from chesspgn.tokens import Token, tokenize

NAGS = {"!": 1, "?": 2, "!!": 3, "??": 4, "!?": 5, "?!": 6}
NAG_EQUAL = 10
BONUS_SAME_FROM_SQUARE = 0.6  # "9 Bb3? ...; 9 Be2": both alternatives move the same piece
BONUS_SAME_PIECE_TYPE = 0.3
BONUS_RECENT_SIBLING = 0.05
PENALTY_NOT_AN_ALTERNATIVE = 5.0  # first move equals the move already played there
_OPEN, _CLOSE = "([", ")]"


@dataclass
class BuiltGame:
    game: chess.pgn.Game
    decoded: Decoded
    flags: list[Flag]  # main-line flags, then side-line flags (reason prefixed "side:")


@dataclass
class _Group:
    alternatives: list[list["Token | _Group"]] = field(default_factory=lambda: [[]])


@dataclass
class _Line:
    start_ply: int  # 0-based ply of the line's first move
    nodes: list[chess.pgn.GameNode]  # nodes[0] = position before the first move
    token_index: list[int]


def build_game(
    text: str, pins: dict[int, chess.Move] | None = None, time_budget_s: float = decoder.DEFAULT_TIME_BUDGET_S
) -> BuiltGame:
    deadline = time.monotonic() + time_budget_s
    tree = _parse_tree(mark_commentary(text))
    tokens, groups = _split(tree.alternatives[0])
    decoded = decoder.decode_tokens(tokens, pins=pins, time_budget_s=time_budget_s)
    decoded.deadline = deadline
    game = chess.pgn.Game()
    game.headers["Result"] = decoded.result
    main = _Line(0, _add_moves(game, decoded.moves, decoded.annotations), decoded.token_index)
    _mark_equal(tokens, main)
    side_flags: list[Flag] = []
    start = decoder.game_start(tokens, decoded.convention)
    for position, group in groups:
        if position > start:  # groups before the game (titles, prose) are not side lines
            _attach(group, position, main, decoded, side_flags)
    return BuiltGame(game, decoded, list(decoded.flags) + side_flags)


# ---------------------------------------------------------------- text -> tree


def _parse_tree(text: str) -> _Group:
    root = _Group()
    stack = [root]
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            stack[-1].alternatives[-1].extend(tokenize("".join(buffer)))
            buffer.clear()

    comment_depth = 0
    for ch in text:
        # { } is a comment (PGN convention; also how layout marks commentary): never a side line.
        if ch == "{":
            if comment_depth == 0:
                flush()
            comment_depth += 1
            continue
        if comment_depth:
            comment_depth -= ch == "}"
            continue
        if ch in _OPEN:
            flush()
            group = _Group()
            stack[-1].alternatives[-1].append(group)
            stack.append(group)
        elif ch in _CLOSE and len(stack) > 1:
            flush()
            stack.pop()
        elif ch == ";" and len(stack) > 1:
            flush()
            stack[-1].alternatives.append([])
        else:
            buffer.append(ch)
    flush()
    return root


def _split(items: list["Token | _Group"]) -> tuple[list[Token], list[tuple[int, _Group]]]:
    """This line's tokens, and each nested group with the number of tokens before it."""
    tokens: list[Token] = []
    groups: list[tuple[int, _Group]] = []
    for item in items:
        if isinstance(item, _Group):
            groups.append((len(tokens), item))
        else:
            tokens.append(item)
    return tokens, groups


# ---------------------------------------------------------------- placement


def _attach(group: _Group, position: int, parent: _Line, decoded: Decoded, flags: list[Flag]) -> None:
    played_before = sum(1 for ti in parent.token_index if ti < position)
    siblings: list[_Line] = []
    for items in group.alternatives:
        tokens, inner_groups = _split(items)
        if not tokens or (tokens[0].number is None and not decoder.looks_like_a_move(tokens[0], decoded.convention)):
            continue  # prose in brackets: a comment, which is dropped
        line = _place(tokens, parent, siblings, played_before, decoded, flags)
        if line is None:
            continue
        siblings.append(line)
        for inner_position, inner in inner_groups:
            _attach(inner, inner_position, line, decoded, flags)


def _place(
    tokens: list[Token], parent: _Line, siblings: list[_Line], played_before: int, decoded: Decoded, flags: list[Flag]
) -> _Line | None:
    first = tokens[0]
    if first.number is not None:
        target = 2 * (first.number - 1) + (1 if first.black_to_move else 0)
    else:
        target = parent.start_ply + played_before - 1  # no number: an alternative to the move just played
    candidates = [(s, BONUS_RECENT_SIBLING) for s in reversed(siblings)] + [(parent, 0.0)]
    best: tuple[float, _Line, int, decoder.Line] | None = None
    for line, bonus in candidates:
        j = target - line.start_ply
        if j < 0 or j >= len(line.nodes):
            continue
        branch = line.nodes[j]
        result = decoder.decode_line(tokens, decoded.convention, branch.board(), decoded.mapping, decoded.deadline)
        if not result.moves:
            continue
        score = result.cost - bonus - 2.0 * len(result.moves) + _replacement_score(line, j, result.moves[0])
        if best is None or score < best[0]:
            best = (score, line, j, result)
    if best is None:
        return None
    _score, line, j, result = best
    nodes = _add_moves(line.nodes[j], result.moves, result.annotations)
    placed = _Line(line.start_ply + j, nodes, result.token_index)
    _mark_equal(tokens, placed)
    # Side-line flags carry the absolute ply (1-based), so the review screen can say which move it is.
    flags.extend(Flag(placed.start_ply + f.ply, f.token, f.chosen, f"side:{f.reason}") for f in result.flags)
    if result.cut_at is not None:
        # The book's own side line has an impossible move here: keep the rest as text.
        cut = result.cut_at - 1 if result.cut_at > 0 and tokens[result.cut_at - 1].number is not None else result.cut_at
        nodes[-1].comment = _raw(tokens[cut:])
    return placed


def _replacement_score(line: _Line, j: int, first: chess.Move) -> float:
    if j + 1 >= len(line.nodes):
        return 0.0
    replaced = line.nodes[j + 1].move
    if first == replaced:
        return PENALTY_NOT_AN_ALTERNATIVE
    board = line.nodes[j].board()
    if first.from_square == replaced.from_square:
        return -BONUS_SAME_FROM_SQUARE
    if board.piece_type_at(first.from_square) == board.piece_type_at(replaced.from_square):
        return -BONUS_SAME_PIECE_TYPE
    return 0.0


# ---------------------------------------------------------------- nodes


def _add_moves(node: chess.pgn.GameNode, moves: list[chess.Move], annotations: list[str]) -> list[chess.pgn.GameNode]:
    nodes = [node]
    for move, note in zip(moves, annotations):
        node = node.add_variation(move)
        if note in NAGS:
            node.nags.add(NAGS[note])
        nodes.append(node)
    return nodes


def _mark_equal(tokens: list[Token], line: _Line) -> None:
    for i, token in enumerate(tokens):
        if token.text == "=":
            before = [n for n, ti in enumerate(line.token_index) if ti < i]
            if before:
                line.nodes[before[-1] + 1].nags.add(NAG_EQUAL)


def _raw(tokens: list[Token]) -> str:
    parts = []
    for t in tokens:
        if t.number is not None:
            parts.append(f"{t.number}{'...' if t.black_to_move else ''}")
        else:
            parts.append(t.text + t.annotation)
    return " ".join(parts)
