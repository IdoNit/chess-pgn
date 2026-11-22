"""Decode a game's main line from noisy book text: search for the cheapest legal game.

Each token may be read as a move (exactly, or after an OCR correction, at a cost), skipped (free for prose,
expensive for something that looks like a move), or replaced by any legal move (expensive: the OCR destroyed
it). A move missing from the text can be inserted. Move numbers and +/x/mate are soft checksums. Every
correction becomes a Flag, so a guess is never silent.
"""

import re
import time
from dataclasses import dataclass, field
from functools import lru_cache

import chess

from chesspgn import notation
from chesspgn.tokens import COST_DROP_GROUP, COST_SUBSTITUTION, RESULT, Token, tokenize, variants

COST_SKIP_MOVELIKE = 3.0  # ignore something that reads as a move
COST_WILDCARD = 4.0  # token unreadable: play any legal move instead
COST_NUMBER_MISMATCH = 1.5
COST_GROUP_CONFLICT = 2.0  # an icon whose group was already learned as another piece
CLOSE_CALL_MARGIN = 1.0  # a different game this close in cost gets a flag where it differs
BEAM_WIDTH = 100
BEAM_WIDTH_OVER_BUDGET = 10
MISSING_MOVE_TOP_STATES = 5
PROBE_TOKENS = 24  # tokens decoded under every convention before committing to one
RETRY_COST_PER_MOVE = 1.0  # a committed convention this costly gets the runner-up tried in full too
DEFAULT_TIME_BUDGET_S = 90.0

_NO_MAPPING: tuple[int, ...] = (0,) * len(notation.GROUP_SYMBOLS)
_MOVEISH = re.compile(r"[1-8x♔♕♖♗♘♙♚♛♜♝♞♟]|^[0O]-[0O]")
_BRACKETS = {"(": ")", "[": "]", "{": "}"}


@dataclass(frozen=True)
class Flag:
    ply: int  # 1-based within its line
    token: str
    chosen: str  # SAN of the move that was played there
    reason: str  # ocr | icon | unreadable | missing | close_call | time_budget (side lines: "side:" prefix)


@dataclass
class Decoded:
    convention: str
    moves: list[chess.Move]
    annotations: list[str]
    token_index: list[int]
    cost: float
    flags: list[Flag]
    result: str
    mapping: tuple[int, ...]
    deadline: float | None = None  # set by build_game: the page's deadline, shared with side lines


@dataclass
class Line:
    """A side line decoded strictly from a given position."""

    moves: list[chess.Move]
    annotations: list[str]
    token_index: list[int]
    cost: float
    cut_at: int | None  # index of the first token the board rejected, else None
    flags: list[Flag]


@dataclass(order=True)
class _State:
    cost: float
    board: chess.Board = field(compare=False)
    moves: tuple[chess.Move, ...] = field(compare=False, default=())
    annotations: tuple[str, ...] = field(compare=False, default=())
    flags: tuple[Flag, ...] = field(compare=False, default=())
    token_index: tuple[int, ...] = field(compare=False, default=())
    mapping: tuple[int, ...] = field(compare=False, default=_NO_MAPPING)
    numbers_in_sync: bool = field(compare=False, default=True)

    def but(self, **changes) -> "_State":
        values = {f: getattr(self, f) for f in self.__dataclass_fields__}
        values.update(changes)
        return _State(**values)


# ---------------------------------------------------------------- public API


def decode_text(
    text: str,
    convention: str | None = None,
    pins: dict[int, chess.Move] | None = None,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
) -> Decoded:
    """Main line of `text`, ignoring everything inside ( ), [ ] and { }."""
    return decode_tokens(tokenize(strip_brackets(text)), convention, pins, time_budget_s)


def decode_tokens(
    tokens: list[Token],
    convention: str | None = None,
    pins: dict[int, chess.Move] | None = None,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
) -> Decoded:
    deadline = time.monotonic() + time_budget_s
    options = {"pins": pins or {}, "deadline": deadline}
    if convention is not None:
        return _decode_with_convention(tokens, convention, game_start(tokens, convention), **options)
    # Probe every convention on the opening moves; a wrong one breaks within a few moves.
    probes = []
    for c in notation.CONVENTIONS:
        start = game_start(tokens, c)
        probes.append((_score(_decode_with_convention(tokens, c, start, start + PROBE_TOKENS, **options)), c, start))
    probes.sort()
    _score_, best_convention, start = probes[0]
    result = _decode_with_convention(tokens, best_convention, start, **options)
    if len(probes) > 1 and result.cost > RETRY_COST_PER_MOVE * max(1, len(result.moves)):
        _s, second, second_start = probes[1]
        other = _decode_with_convention(tokens, second, second_start, **options)
        result = min((result, other), key=_score)
    return result


def _score(decoded: "Decoded") -> float:
    return decoded.cost - 2 * len(decoded.moves)


def strip_brackets(text: str) -> str:
    out: list[str] = []
    stack: list[str] = []
    for ch in text:
        if ch in _BRACKETS:
            stack.append(_BRACKETS[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
            out.append(" ")
        elif not stack:
            out.append(ch)
    return "".join(out)


def game_start(tokens: list[Token], convention: str) -> int:
    """Index of the first "1" followed by a legal first move; earlier text (prose, tables) is ignored."""
    for exact in (True, False):
        for i, token in enumerate(tokens[:-1]):
            is_one = (token.number == 1 and not token.black_to_move) or token.text in OCR_ONES
            if is_one and _first_move_cost(tokens[i + 1].text, convention) <= (0.0 if exact else COST_SUBSTITUTION):
                return i
    return 0


OCR_ONES = {"l", "I", "l.", "I.", "|"}


def is_first_move(text: str) -> bool:
    """Whether `text` reads, in some convention and with at most one OCR correction, as a legal first move."""
    return any(_first_move_cost(text, c) <= COST_SUBSTITUTION for c in notation.CONVENTIONS)


def _first_move_cost(text: str, convention: str) -> float:
    start_board = chess.Board()
    readings = _readings(text, convention)
    costs = [found[0] for m in start_board.legal_moves if (found := _reading_match(start_board, m, readings, _NO_MAPPING))]
    return min(costs, default=float("inf"))


def decode_line(
    tokens: list[Token], convention: str, board: chess.Board, mapping: tuple[int, ...], deadline: float | None = None
) -> Line:
    """Decode a side line from `board`. Strict: a move is never skipped or invented; stop at the first
    move the board rejects. `deadline` is the page's (time.monotonic()) deadline."""
    if deadline is None:
        deadline = time.monotonic() + DEFAULT_TIME_BUDGET_S
    beam, _budget_ply, cut_at = _search(tokens, 0, convention, board, {}, deadline, strict=True, mapping=mapping)
    best = beam[0]
    return Line(list(best.moves), list(best.annotations), list(best.token_index), best.cost, cut_at, list(best.flags))


def looks_like_a_move(token: Token, convention: str) -> bool:
    if token.number is not None or not token.text:
        return False
    return any(group[0][1] <= COST_SUBSTITUTION for group in _readings(token.text, convention).values())


# ---------------------------------------------------------------- search


def _decode_with_convention(
    tokens: list[Token],
    convention: str,
    start: int,
    stop: int | None = None,
    *,
    pins: dict[int, chess.Move],
    deadline: float,
) -> Decoded:
    end = len(tokens) if stop is None else min(stop, len(tokens))
    beam, budget_ply, _cut = _search(tokens[:end], start, convention, chess.Board(), pins, deadline)
    decoded = _finish(beam, convention, _result(tokens[start:end], beam[0]))
    if budget_ply is not None:
        decoded.flags.append(Flag(min(budget_ply, max(1, len(decoded.moves))), "", "", "time_budget"))
        decoded.flags.sort(key=lambda f: f.ply)
    return decoded


def _search(
    tokens: list[Token],
    start: int,
    convention: str,
    board: chess.Board,
    pins: dict[int, chess.Move],
    deadline: float,
    *,
    strict: bool = False,
    mapping: tuple[int, ...] = _NO_MAPPING,
) -> tuple[list[_State], int | None, int | None]:
    """Final beam; the ply from which the time budget forced a narrow search (None if never); and in
    strict mode the index of the token where decoding stopped (None if it reached the end)."""
    beam = [_State(0.0, board.copy(stack=False), mapping=mapping)]
    budget_ply: int | None = None
    for index in range(start, len(tokens)):
        token = tokens[index]
        if budget_ply is None and time.monotonic() > deadline:
            budget_ply = len(beam[0].moves) + 1
        if token.number is not None:
            beam = [_apply_number(s, token) for s in beam]
        elif RESULT.match(token.text) is not None:
            if not strict:
                break  # the game is over; anything after it belongs to the next game
        elif token.text != "=":
            new_beam, unmatched_move = _expand(
                beam, token, index, convention, pins, hurried=budget_ply is not None, strict=strict
            )
            if strict and unmatched_move:
                return beam, budget_ply, index
            beam = new_beam
    return beam, budget_ply, None


def _apply_number(state: _State, token: Token) -> _State:
    black_to_move = state.board.turn == chess.BLACK
    if state.board.fullmove_number == token.number and black_to_move == token.black_to_move:
        return state if state.numbers_in_sync else state.but(numbers_in_sync=True)
    flags = state.flags
    if state.numbers_in_sync:  # flag the first mismatch of a run; numbers are OCR'd and misprinted too
        flags = flags + (Flag(len(state.moves) + 1, str(token.number), "", "number"),)
    return state.but(cost=state.cost + COST_NUMBER_MISMATCH, flags=flags, numbers_in_sync=False)


def _expand(
    beam: list[_State],
    token: Token,
    index: int,
    convention: str,
    pins: dict[int, chess.Move],
    *,
    hurried: bool,
    strict: bool,
) -> tuple[list[_State], bool]:
    """New beam, and whether this token looked like a move that no state could play."""
    readings = _readings(token.text, convention)
    near_reading = any(group[0][1] <= COST_SUBSTITUTION for group in readings.values())
    junk = sum(not ch.isalnum() and ch not in "-+/" for ch in token.text) * 2 >= len(token.text)
    # "Nb&", "Nbe": a piece letter followed by file letters/junk is a damaged move; "Black", "Both" are words.
    piece_led = (
        token.text[:1] in _piece_starts(convention)
        and len(token.text) <= 5
        and not re.search(r"[i-z]", token.text[1:])
    )
    looks_like_move = near_reading or (len(token.text) <= 7 and (bool(_MOVEISH.search(token.text)) or junk or piece_led))
    best: dict[tuple, _State] = {}

    def offer(state: _State) -> None:
        key = (state.board._transposition_key(), len(state.moves), state.mapping)
        if key not in best or state.cost < best[key].cost:
            best[key] = state

    any_matched = False
    for rank, state in enumerate(beam):
        if not looks_like_move:
            offer(state.but())
        elif not strict:  # a side line never skips a move: it matches or is cut
            skipped = Flag(len(state.moves) + 1, token.text, "", "skipped")
            offer(state.but(cost=state.cost + COST_SKIP_MOVELIKE, flags=state.flags + (skipped,)))
        pinned = pins.get(len(state.moves) + 1)
        if pinned is not None and state.board.is_legal(pinned):  # an illegal (stale) pin is ignored here
            if looks_like_move:
                offer(_play(state, pinned, 0.0, token, "", index, state.mapping))
                any_matched = True
            continue
        matched, matched_cleanly = False, False
        for move in state.board.legal_moves:
            found = _reading_match(state.board, move, readings, state.mapping)
            if found is not None:
                matched = True
                matched_cleanly = matched_cleanly or found[0] < COST_SUBSTITUTION
                offer(_play(state, move, found[0], token, _reason(found[0]), index, found[1]))
        any_matched = any_matched or matched
        if strict:
            continue
        if looks_like_move and not matched_cleanly:
            for move in state.board.legal_moves:
                offer(_play(state, move, COST_WILDCARD, token, "unreadable", index, state.mapping))
            if rank < MISSING_MOVE_TOP_STATES and not hurried:
                _offer_missing_move(state, token, readings, index, offer)
    width = BEAM_WIDTH_OVER_BUDGET if hurried else BEAM_WIDTH
    return sorted(best.values())[:width], looks_like_move and not any_matched


@lru_cache(maxsize=None)
def _piece_starts(convention: str) -> frozenset[str]:
    letters = {"K", "Q", "R", "B", "N", "P"} if convention == "descriptive" else {k[0] for k in notation.PIECE_BY_LETTER[convention]}
    return frozenset(letters | set(notation.FIGURINES) | set(notation.GROUP_SYMBOLS))


def _offer_missing_move(state: _State, token: Token, readings, index: int, offer) -> None:
    """The book's previous move is missing from the text: insert one, then read this token."""
    for missing in state.board.legal_moves:
        inserted = _play(state, missing, COST_WILDCARD, Token("<missing>"), "missing", index, state.mapping)
        for move in inserted.board.legal_moves:
            found = _reading_match(inserted.board, move, readings, inserted.mapping)
            if found is not None:
                offer(_play(inserted, move, found[0], token, _reason(found[0]), index, found[1]))


def _reason(cost: float) -> str:
    if cost >= COST_SUBSTITUTION:
        return "ocr"
    if cost >= COST_DROP_GROUP:
        return "icon"
    return ""


def _play(
    state: _State, move: chess.Move, cost: float, token: Token, reason: str, index: int, mapping: tuple[int, ...]
) -> _State:
    board = state.board.copy(stack=False)
    flags = state.flags
    if reason:
        flags = flags + (Flag(len(state.moves) + 1, token.text, board.san(move), reason),)
    board.push(move)
    return _State(
        state.cost + cost, board, state.moves + (move,), state.annotations + (token.annotation,), flags,
        state.token_index + (index,), mapping, state.numbers_in_sync,
    )


# ---------------------------------------------------------------- reading tokens


@lru_cache(maxsize=400_000)
def _readings(text: str, convention: str) -> dict[object, tuple[tuple[notation.Clues, float], ...]]:
    """Readings of a token grouped by moving piece ("castle", "group" for icon groups), cheapest first."""
    groups: dict[object, list[tuple[notation.Clues, float]]] = {}
    for variant, cost in variants(text):
        clues = notation.read_clues(variant, convention)
        if clues is None:
            continue
        if clues.castle is not None:
            key: object = "castle"
        elif clues.piece_class is not None:
            key = "group"
        else:
            key = clues.piece
        groups.setdefault(key, []).append((clues, cost))
    return {k: tuple(v) for k, v in groups.items()}


def _reading_match(
    board: chess.Board, move: chess.Move, readings, mapping: tuple[int, ...]
) -> tuple[float, tuple[int, ...]] | None:
    """Cheapest reading of the token that fits `move`, and the icon-group mapping that results."""
    piece = board.piece_type_at(move.from_square)
    key: object = "castle" if board.is_castling(move) else piece
    candidates = list(readings.get(key, ()))
    if key not in ("castle", chess.PAWN):
        candidates += readings.get("group", ())
    best: tuple[float, tuple[int, ...]] | None = None
    for clues, variant_cost in sorted(candidates, key=lambda r: r[1]):
        if best is not None and variant_cost >= best[0]:
            break
        penalty = notation.match_cost(board, move, clues)
        if penalty is None:
            continue
        new_mapping = mapping
        if clues.piece_class is not None:
            learned = mapping[clues.piece_class]
            if learned == 0:
                new_mapping = mapping[: clues.piece_class] + (piece,) + mapping[clues.piece_class + 1:]
            elif learned != piece:
                penalty += COST_GROUP_CONFLICT
        total = variant_cost + penalty
        if best is None or total < best[0]:
            best = (total, new_mapping)
    return best


# ---------------------------------------------------------------- result


def _finish(beam: list[_State], convention: str, result: str) -> Decoded:
    best = beam[0]
    flags = list(best.flags)
    runner_up = next((s for s in beam[1:] if len(s.moves) == len(best.moves) and s.moves != best.moves), None)
    if runner_up is not None and runner_up.cost - best.cost < CLOSE_CALL_MARGIN:
        ply = next(i for i, (a, b) in enumerate(zip(best.moves, runner_up.moves), start=1) if a != b)
        if not any(f.ply == ply for f in flags):
            board = chess.Board()
            for move in best.moves[: ply - 1]:
                board.push(move)
            flags.append(Flag(ply, "", board.san(best.moves[ply - 1]), "close_call"))
    return Decoded(
        convention=convention,
        moves=list(best.moves),
        annotations=list(best.annotations),
        token_index=list(best.token_index),
        cost=best.cost,
        flags=sorted(flags, key=lambda f: f.ply),
        result=result,
        mapping=best.mapping,
    )


def _result(tokens: list[Token], best: _State) -> str:
    for token in tokens:
        if RESULT.match(token.text):
            return {"1:0": "1-0", "0:1": "0-1", "½-½": "1/2-1/2"}.get(token.text, token.text)
    words = " ".join(t.text.lower() for t in tokens[-6:])
    if "black resigned" in words or "black resigns" in words:
        return "1-0"
    if "white resigned" in words or "white resigns" in words:
        return "0-1"
    if re.search(r"\bresign(s|ed)?\b", words):  # a bare "Resigns": the side that just moved won
        return "0-1" if best.board.turn == chess.WHITE else "1-0"
    if "and won" in words or "and wins" in words or best.board.is_checkmate():
        return "0-1" if best.board.turn == chess.WHITE else "1-0"  # the side that moved last won
    return "*"


def suggest_moves(board: chess.Board, token_text: str, chosen: chess.Move, limit: int = 3) -> list[str]:
    """Other legal moves that could be what the book printed, best fit first (SAN)."""
    scored: list[tuple[float, str]] = []
    for move in board.legal_moves:
        if move == chosen:
            continue
        fits = [
            found[0]
            for convention in notation.CONVENTIONS
            if token_text
            and (found := _reading_match(board, move, _readings(token_text, convention), _NO_MAPPING)) is not None
        ]
        if fits:
            score = min(fits)
        elif move.to_square == chosen.to_square:
            score = 5.0
        elif board.piece_type_at(move.from_square) == board.piece_type_at(chosen.from_square):
            score = 6.0
        else:
            continue
        scored.append((score, board.san(move)))
    return [san for _score, san in sorted(scored)[:limit]]
