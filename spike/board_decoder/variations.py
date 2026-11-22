"""Build a full PGN game (main line + side lines) from book text.

Structure comes from the text ([ ] / ( ) groups, ';' separating alternatives); *placement* comes from the
board: each alternative is tried at every line where its move number fits, and kept where its moves are legal.
"""

import io
from dataclasses import dataclass, field

import chess
import chess.pgn

import decoder
from decoder import Token

_NAGS = {"!": chess.pgn.NAG_GOOD_MOVE, "?": chess.pgn.NAG_MISTAKE, "!!": chess.pgn.NAG_BRILLIANT_MOVE,
         "??": chess.pgn.NAG_BLUNDER, "!?": chess.pgn.NAG_SPECULATIVE_MOVE, "?!": chess.pgn.NAG_DUBIOUS_MOVE}
NAG_EQUAL = 10
_OPEN, _CLOSE = "([{", ")]}"

# Placement tie-breaks (lower is better), used only between alternatives that are all legal.
BONUS_SAME_FROM_SQUARE = 0.6  # "9 Bb3? ...; 9 Be2": both moves are by the same piece
BONUS_SAME_PIECE_TYPE = 0.3
BONUS_RECENT_SIBLING = 0.05
PENALTY_NOT_AN_ALTERNATIVE = 5.0  # first move equals the move already played there


# ---------------------------------------------------------------- text -> tree


@dataclass
class Group:
    alternatives: list[list["Token | Group"]] = field(default_factory=lambda: [[]])


def parse_tree(text: str) -> Group:
    root = Group()
    stack = [root]
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            stack[-1].alternatives[-1].extend(decoder.tokenize("".join(buffer)))
            buffer.clear()

    for ch in text:
        if ch in _OPEN:
            flush()
            group = Group()
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


def _split(items: list["Token | Group"]) -> tuple[list[Token], list[tuple[int, Group]]]:
    """Tokens of this line, and each nested group with the number of tokens that precede it."""
    tokens: list[Token] = []
    groups: list[tuple[int, Group]] = []
    for item in items:
        if isinstance(item, Group):
            groups.append((len(tokens), item))
        else:
            tokens.append(item)
    return tokens, groups


# ---------------------------------------------------------------- lines


@dataclass
class _Line:
    start_ply: int  # ply of the first move of this line (0 = White's 1st move)
    nodes: list[chess.pgn.GameNode]  # nodes[0] = position before the first move
    token_index: list[int]  # token index of each move


def _ply_of(token: Token) -> int:
    return 2 * (token.number - 1) + (1 if token.black_to_move else 0)


def _add_moves(node: chess.pgn.GameNode, moves: list[chess.Move], annotations: list[str]) -> list[chess.pgn.GameNode]:
    nodes = [node]
    for move, note in zip(moves, annotations):
        node = node.add_variation(move)
        if note in _NAGS:
            node.nags.add(_NAGS[note])
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


# ---------------------------------------------------------------- build


def build_game(text: str, convention: str | None = None) -> tuple[chess.pgn.Game, decoder.Decoded]:
    tree = parse_tree(text)
    main_tokens, main_groups = _split(tree.alternatives[0])
    decoded = decoder.decode_tokens(main_tokens, convention)
    game = chess.pgn.Game()
    game.headers["Result"] = decoded.result
    main = _Line(0, _add_moves(game, decoded.moves, decoded.annotations), decoded.token_index)
    _mark_equal(main_tokens, main)
    start = decoder.game_start(main_tokens, decoded.convention)
    for position, group in main_groups:
        if position > start:  # groups before the game (titles, prose) are not side lines
            _attach_group(group, position, main, decoded.convention, decoded.mapping, decoded.side_line_repairs)
    return game, decoded


def _attach_group(
    group: Group, position: int, parent: _Line, convention: str, mapping: tuple[int, ...], flags: list[decoder.Repair]
) -> None:
    played_before = sum(1 for ti in parent.token_index if ti < position)
    siblings: list[_Line] = []
    for items in group.alternatives:
        tokens, inner_groups = _split(items)
        if not tokens or (tokens[0].number is None and not decoder.looks_like_a_move(tokens[0], convention)):
            continue  # prose in brackets: a comment, which we drop
        line = _place(tokens, parent, siblings, played_before, convention, mapping, flags)
        if line is None:
            continue
        siblings.append(line)
        for inner_position, inner in inner_groups:
            _attach_group(inner, inner_position, line, convention, mapping, flags)


def _place(
    tokens: list[Token], parent: _Line, siblings: list[_Line], played_before: int, convention: str,
    mapping: tuple[int, ...], flags: list[decoder.Repair],
) -> _Line | None:
    target = _ply_of(tokens[0]) if tokens[0].number is not None else parent.start_ply + played_before - 1
    candidates = [(s, BONUS_RECENT_SIBLING) for s in reversed(siblings)] + [(parent, 0.0)]
    best: tuple[float, _Line, int, decoder.Line] | None = None
    for line, bonus in candidates:
        j = target - line.start_ply
        if j < 0 or j >= len(line.nodes):
            continue
        branch = line.nodes[j]
        decoded = decoder.decode_line(tokens, convention, branch.board(), mapping)
        if not decoded.moves:
            continue
        score = decoded.cost - bonus - 2.0 * len(decoded.moves)
        if j + 1 < len(line.nodes):
            replaced = line.nodes[j + 1].move
            first = decoded.moves[0]
            if first == replaced:
                score += PENALTY_NOT_AN_ALTERNATIVE
            board = branch.board()
            if first.from_square == replaced.from_square:
                score -= BONUS_SAME_FROM_SQUARE
            elif board.piece_type_at(first.from_square) == board.piece_type_at(replaced.from_square):
                score -= BONUS_SAME_PIECE_TYPE
        if best is None or score < best[0]:
            best = (score, line, j, decoded)
    if best is None:
        return None
    _score, line, j, decoded = best
    nodes = _add_moves(line.nodes[j], decoded.moves, decoded.annotations)
    placed = _Line(line.start_ply + j, nodes, decoded.token_index)
    flags.extend(decoded.repairs)
    _mark_equal(tokens, placed)
    if decoded.cut_at is not None:
        # The book's own side line has an impossible move here: keep the rest as text.
        cut = decoded.cut_at - 1 if decoded.cut_at > 0 and tokens[decoded.cut_at - 1].number is not None else decoded.cut_at
        nodes[-1].comment = (nodes[-1].comment + " " if nodes[-1].comment else "") + _raw(tokens[cut:])
    return placed


def to_pgn(game: chess.pgn.Game) -> str:
    return str(game)


def read_pgn(text: str) -> chess.pgn.Game:
    return chess.pgn.read_game(io.StringIO(text))


def all_variation_moves(game: chess.pgn.Game) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    stack: list[tuple[chess.pgn.GameNode, bool]] = [(game, True)]
    while stack:
        node, on_main = stack.pop()
        for i, child in enumerate(node.variations):
            child_main = on_main and i == 0
            if not child_main:
                found.add((node.board().epd(), child.move.uci()))
            stack.append((child, child_main))
    return found

