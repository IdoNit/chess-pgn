"""Decode the main line of a game from noisy book text by searching for the cheapest legal game.

Every token may be: read as a move (exactly, or after an OCR-style correction, at a cost),
skipped (cheap for prose, expensive for something that looks like a move), or replaced by an
arbitrary legal move (expensive: for a token the OCR destroyed). Move numbers and +/x/mate are
soft checksums. The search keeps the best few hundred partial games and returns the cheapest
complete one, plus every place where it had to repair something.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

import chess

import notation

# ---------------------------------------------------------------- costs
COST_SUBSTITUTION = 1.0  # one OCR-confusable character changed
COST_DELETION = 1.5  # one character the OCR invented
COST_INSERTION = 1.2  # one character the OCR lost
COST_DROP_GROUP = 0.8  # an icon group symbol attached to the wrong move
_INSERTABLE = "KQRBNP12345678abcdefgh"
COST_SKIP_MOVELIKE = 3.0  # ignore something that reads as a legal-looking move
COST_SKIP_TEXT = 0.0
COST_WILDCARD = 4.0  # token unreadable: play any legal move instead
COST_NUMBER_MISMATCH = 1.5
BEAM_WIDTH = 100
MISSING_MOVE_TOP_STATES = 5

# Characters an OCR engine commonly confuses, both directions.
_CONFUSIONS: list[tuple[str, str]] = [
    ("S", "5"), ("O", "0"), ("l", "1"), ("I", "1"), ("B", "8"), ("Z", "2"), ("g", "9"),
    ("b", "6"), ("e", "c"), ("h", "b"), ("K", "R"), ("N", "H"), ("B", "R"), ("Q", "O"),
    ("x", "X"), ("-", "x"), ("1", "7"), ("3", "8"), ("6", "5"), ("♗", "♕"), ("♖", "♔"), ("N", "M"),
]
_SUBS: dict[str, set[str]] = {}
for a, b in _CONFUSIONS:
    _SUBS.setdefault(a, set()).add(b)
    _SUBS.setdefault(b, set()).add(a)

_DASHES = re.compile(r"[–—−‒]+|-{2,}")
_NUMBER = re.compile(r"^(?P<n>\d{1,3})(?P<dots>\.{1,3}|…)?$")
_NUMBER_GLUED = re.compile(r"^(?P<n>\d{1,3})(?P<dots>\.{1,3}|…)(?P<rest>\S+)$")
_ANNOTATION = re.compile(r"[!?]+$")
_RESULT = re.compile(r"^(1-0|0-1|1:0|0:1|½-½|1/2-1/2)$")
_MOVEISH = re.compile(r"[1-8x♔♕♖♗♘♙♚♛♜♝♞♟]|^[0O]-[0O]")


@dataclass(frozen=True)
class Token:
    text: str
    annotation: str = ""
    number: int | None = None
    black_to_move: bool = False


@dataclass
class Repair:
    token: str
    chosen: str
    reason: str
    ply: int


@dataclass
class Decoded:
    convention: str
    moves: list[chess.Move]
    annotations: list[str]
    cost: float
    repairs: list[Repair]
    result: str
    runner_up_cost: float | None
    runner_up_differs_at: int | None
    token_index: list[int] = field(default_factory=list)
    tokens: list[Token] = field(default_factory=list)
    mapping: tuple[int, ...] = (0,) * 8
    side_line_repairs: list[Repair] = field(default_factory=list)


@dataclass(order=True)
class _State:
    cost: float
    board: chess.Board = field(compare=False)
    moves: tuple[chess.Move, ...] = field(compare=False, default=())
    annotations: tuple[str, ...] = field(compare=False, default=())
    repairs: tuple[Repair, ...] = field(compare=False, default=())
    token_index: tuple[int, ...] = field(compare=False, default=())
    mapping: tuple[int, ...] = field(compare=False, default=(0,) * 8)  # icon group -> piece type (0 = unknown)


_NO_MAPPING: tuple[int, ...] = (0,) * 8
COST_GROUP_CONFLICT = 2.0  # an icon whose group was already learned as another piece (misgrouped icon)


# ---------------------------------------------------------------- text -> tokens


def strip_comments(text: str) -> tuple[str, list[str]]:
    """Remove (...) and [...] groups. Returned separately; the main line is decoded without them."""
    removed: list[str] = []
    out: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch in "([{":
            depth += 1
            if depth == 1:
                current = []
                continue
        if ch in ")]}" and depth > 0:
            depth -= 1
            if depth == 0:
                removed.append("".join(current))
                out.append(" ")
                continue
        (current if depth else out).append(ch)
    return "".join(out), removed


def tokenize(text: str) -> list[Token]:
    text = _DASHES.sub("-", text).replace("…", "...")
    text = re.sub(r"\.\s+\.\s*\.", "...", text)  # "16 . . . N-Q2"
    tokens: list[Token] = []
    for raw in text.split():
        raw = raw.strip(",;:")  # keep "1:0" intact below
        if not raw:
            continue
        glued = _NUMBER_GLUED.match(raw)
        if glued and not _RESULT.match(raw):
            tokens.append(Token("", number=int(glued["n"]), black_to_move=len(glued["dots"]) >= 2 or glued["dots"] == "…"))
            raw = glued["rest"]
        number = _NUMBER.match(raw)
        if number:
            dots = number["dots"] or ""
            tokens.append(Token("", number=int(number["n"]), black_to_move=len(dots) >= 2))
            continue
        if raw in ("...", ".."):
            continue
        annotation_match = _ANNOTATION.search(raw)
        annotation = annotation_match.group(0) if annotation_match else ""
        body = raw[: annotation_match.start()] if annotation_match else raw
        if body:
            tokens.append(Token(body, annotation=annotation))
    return tokens


# ---------------------------------------------------------------- variants


@lru_cache(maxsize=200_000)
def _variants(text: str) -> tuple[tuple[str, float], ...]:
    """(variant, cost): the token itself, then 1-2 OCR corrections."""
    found: dict[str, float] = {text: 0.0}
    for variant, cost in list(_one_edit(text)):
        found.setdefault(variant, cost)
    for variant, cost in list(found.items()):
        if 0 < cost <= COST_SUBSTITUTION:
            for v2, c2 in _one_edit(variant):
                found.setdefault(v2, cost + c2)
    for variant, cost in _insertions(text):  # single layer only: keeps the variant count bounded
        found.setdefault(variant, cost)
    return tuple(sorted(found.items(), key=lambda kv: kv[1]))


def _one_edit(text: str):
    if text[:1] in notation.CLASS_SYMBOLS:  # icon linked to the wrong move: drop it (cheapest explanation)
        yield text[1:], COST_DROP_GROUP
    for i, ch in enumerate(text):
        for sub in _SUBS.get(ch, ()):
            yield text[:i] + sub + text[i + 1 :], COST_SUBSTITUTION
        if len(text) > 2:
            yield text[:i] + text[i + 1 :], COST_DELETION
    if "x" not in text and len(text) >= 2:  # dropped capture sign: "PP" -> "PxP", "Nf6" -> "Nxf6"
        for i in range(1, len(text)):
            yield text[:i] + "x" + text[i:], COST_SUBSTITUTION


def _insertions(text: str):
    """A character the OCR lost entirely ("xN" -> "RxN", "Qx" -> "QxP", "KR-K" -> "KR-K1")."""
    for i in range(len(text) + 1):
        for ch in _INSERTABLE:
            yield text[:i] + ch + text[i:], COST_INSERTION


@lru_cache(maxsize=400_000)
def _readings(text: str, convention: str) -> dict[object, tuple[tuple[notation.Clues, float], ...]]:
    """Readings of a token grouped by moving piece ("castle" for castling), each group sorted by cost."""
    groups: dict[object, list[tuple[notation.Clues, float]]] = {}
    for variant, cost in _variants(text):
        clues = notation.read_clues(variant, convention)
        if clues is not None:
            key = "castle" if clues.castle else ("group" if clues.piece_class is not None else clues.piece)
            groups.setdefault(key, []).append((clues, cost))
    return {k: tuple(sorted(v, key=lambda r: r[1])) for k, v in groups.items()}


# ---------------------------------------------------------------- search


def decode(text: str, convention: str | None = None) -> Decoded:
    main_text, _comments = strip_comments(text)
    return decode_tokens(tokenize(main_text), convention)


def decode_tokens(tokens: list[Token], convention: str | None = None) -> Decoded:
    if convention is not None:
        return _decode_tokens(tokens, convention)
    # Try every convention; a wrong one breaks within a few moves and costs far more.
    results = [_decode_tokens(tokens, c) for c in notation.CONVENTIONS]
    return min(results, key=lambda r: (r.cost - 2 * len(r.moves)))


def looks_like_a_move(token: Token, convention: str) -> bool:
    if token.number is not None or not token.text:
        return False
    return any(group[0][1] <= COST_SUBSTITUTION for group in _readings(token.text, convention).values())


def game_start(tokens: list[Token], convention: str) -> int:
    """Index of the first "1" followed by a legal first move; text before it (prose, tables) is ignored."""
    start_board = chess.Board()
    for i, token in enumerate(tokens[:-1]):
        if token.number != 1 or token.black_to_move:
            continue
        readings = _readings(tokens[i + 1].text, convention)
        if any(_best_reading_cost(start_board, m, readings) == 0.0 for m in start_board.legal_moves):
            return i
    return 0


def _decode_tokens(tokens: list[Token], convention: str) -> Decoded:
    beam, result, _cut = _search(tokens, game_start(tokens, convention), convention, chess.Board(), strict=False)
    result = _result_from_words(tokens, beam[0] if beam else None, result)
    return _finish(beam, convention, result)


@dataclass
class Line:
    moves: list[chess.Move]
    annotations: list[str]
    token_index: list[int]
    cost: float
    cut_at: int | None  # index of the first token the board rejected (strict mode), else None
    repairs: list[Repair] = field(default_factory=list)


def decode_line(tokens: list[Token], convention: str, board: chess.Board, mapping: tuple[int, ...] = _NO_MAPPING) -> Line:
    """Decode a side line from `board`. Strict: no invented moves; stops at the first impossible move."""
    beam, _result, cut = _search(tokens, 0, convention, board, strict=True, mapping=mapping)
    best = beam[0]
    return Line(list(best.moves), list(best.annotations), list(best.token_index), best.cost, cut, list(best.repairs))


def _search(
    tokens: list[Token], start: int, convention: str, board: chess.Board, *, strict: bool,
    mapping: tuple[int, ...] = _NO_MAPPING,
):
    beam: list[_State] = [_State(0.0, board.copy(stack=False), mapping=mapping)]
    result = "*"
    for index in range(start, len(tokens)):
        token = tokens[index]
        if token.number is not None:
            beam = [_apply_number(s, token) for s in beam]
            continue
        if _RESULT.match(token.text):
            result = _normalize_result(token.text)
            continue
        if token.text == "=":
            continue
        new_beam, unmatched_move = _expand(beam, token, convention, index, strict)
        if strict and unmatched_move:
            return beam, result, index
        beam = new_beam
    return beam, result, None


def _apply_number(state: _State, token: Token) -> _State:
    expected_black = state.board.turn == chess.BLACK
    if state.board.fullmove_number == token.number and expected_black == token.black_to_move:
        return state
    # Numbers are OCR'd too, and books skip/misprint them: soft penalty only.
    return _State(state.cost + COST_NUMBER_MISMATCH, state.board, state.moves, state.annotations, state.repairs, state.token_index, state.mapping)


def _expand(
    beam: list[_State], token: Token, convention: str, index: int, strict: bool
) -> tuple[list[_State], bool]:
    """New beam, and whether this token was an unmatched move (strict mode stops there)."""
    any_state_matched = False
    readings = _readings(token.text, convention)
    junk = sum(not ch.isalnum() and ch not in "-+/" for ch in token.text) * 2 >= len(token.text)
    # A token is a (possibly damaged) move if one OCR correction away from a valid move.
    near_reading = any(group[0][1] <= COST_SUBSTITUTION for group in readings.values())
    looks_like_move = near_reading or (
        len(token.text) <= 7 and (bool(_MOVEISH.search(token.text)) or junk)
    )
    best: dict[str, _State] = {}

    def offer(state: _State) -> None:
        key = (state.board._transposition_key(), len(state.moves), state.mapping)
        if key not in best or state.cost < best[key].cost:
            best[key] = state

    for rank, state in enumerate(beam):
        if not (strict and looks_like_move):  # a side line never skips a move: it matches or is cut
            skip_cost = COST_SKIP_MOVELIKE if looks_like_move else COST_SKIP_TEXT
            offer(_State(state.cost + skip_cost, state.board, state.moves, state.annotations, state.repairs, state.token_index, state.mapping))
        matched_any = False
        for move in state.board.legal_moves:
            found = _reading_match(state.board, move, readings, state.mapping)
            if found is None:
                continue
            cost, mapping = found
            matched_any = True
            offer(_play(state, move, cost, token, _repair_reason(cost), index, mapping))
        any_state_matched = any_state_matched or matched_any
        if strict:
            continue
        if looks_like_move and not matched_any:
            for move in state.board.legal_moves:
                offer(_play(state, move, COST_WILDCARD, token, "unreadable token, chosen by later moves", index, state.mapping))
            if rank < MISSING_MOVE_TOP_STATES:
                _offer_missing_move(state, token, readings, offer, index)

    ranked = sorted(best.values())
    return ranked[:BEAM_WIDTH], looks_like_move and not any_state_matched


def _offer_missing_move(state: _State, token: Token, readings, offer, index: int) -> None:
    """The book's previous move is missing from the text: insert one, then read this token."""
    for missing in state.board.legal_moves:
        inserted = _play(
            state, missing, COST_WILDCARD, Token("<missing>"), "move missing from text, chosen by later moves", index, state.mapping
        )
        for move in inserted.board.legal_moves:
            found = _reading_match(inserted.board, move, readings, inserted.mapping)
            if found is not None:
                cost, mapping = found
                offer(_play(inserted, move, cost, token, _repair_reason(cost), index, mapping))


def _repair_reason(cost: float) -> str:
    """Every reading that needed a correction is reported, so a guess is never silent."""
    if cost >= COST_SUBSTITUTION:
        return "OCR correction"
    if cost >= COST_DROP_GROUP:
        return "icon ignored here: it may belong to a neighbouring move"
    return ""


def _best_reading_cost(board: chess.Board, move: chess.Move, readings) -> float | None:
    found = _reading_match(board, move, readings, _NO_MAPPING)
    return None if found is None else found[0]


def _reading_match(
    board: chess.Board, move: chess.Move, readings, mapping: tuple[int, ...]
) -> tuple[float, tuple[int, ...]] | None:
    """Cheapest reading of the token that fits `move`, and the icon-group mapping that results."""
    piece = board.piece_type_at(move.from_square)
    key = "castle" if board.is_castling(move) else piece
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
                new_mapping = mapping[: clues.piece_class] + (piece,) + mapping[clues.piece_class + 1 :]
            elif learned != piece:
                penalty += COST_GROUP_CONFLICT
        total = variant_cost + penalty
        if best is None or total < best[0]:
            best = (total, new_mapping)
    return best


def _play(
    state: _State, move: chess.Move, cost: float, token: Token, reason: str, index: int, mapping: tuple[int, ...]
) -> _State:
    board = state.board.copy(stack=False)
    repairs = state.repairs
    if reason:
        repairs = repairs + (Repair(token.text, board.san(move), reason, len(state.moves) + 1),)
    board.push(move)
    return _State(
        state.cost + cost, board, state.moves + (move,), state.annotations + (token.annotation,), repairs,
        state.token_index + (index,), mapping,
    )


def _finish(beam: list[_State], convention: str, result: str) -> Decoded:
    if not beam:
        return Decoded(convention, [], [], float("inf"), [], result, None, None)
    best = beam[0]
    runner_up = next((s for s in beam[1:] if s.moves != best.moves and len(s.moves) == len(best.moves)), None)
    differs_at = None
    if runner_up is not None:
        differs_at = next(i for i, (a, b) in enumerate(zip(best.moves, runner_up.moves)) if a != b) + 1
    return Decoded(
        convention=convention,
        moves=list(best.moves),
        annotations=list(best.annotations),
        cost=best.cost,
        repairs=list(best.repairs),
        result=result,
        runner_up_cost=runner_up.cost if runner_up else None,
        runner_up_differs_at=differs_at,
        token_index=list(best.token_index),
        mapping=best.mapping,
    )


def _normalize_result(text: str) -> str:
    return {"1:0": "1-0", "0:1": "0-1", "½-½": "1/2-1/2"}.get(text, text)


def _result_from_words(tokens: list[Token], best: _State | None, current: str) -> str:
    if current != "*":
        return current
    words = " ".join(t.text.lower() for t in tokens[-6:])
    if "black resigned" in words or "black resigns" in words:
        return "1-0"
    if "white resigned" in words or "white resigns" in words:
        return "0-1"
    if ("and won" in words or "and wins" in words) and best is not None:
        # The side that made the last move won.
        return "0-1" if best.board.turn == chess.WHITE else "1-0"
    if best is not None and best.board.is_checkmate():
        return "0-1" if best.board.turn == chess.WHITE else "1-0"
    return current


def to_pgn_movetext(decoded: Decoded) -> str:
    board = chess.Board()
    parts: list[str] = []
    for move, note in zip(decoded.moves, decoded.annotations):
        if board.turn == chess.WHITE:
            parts.append(f"{board.fullmove_number}.")
        parts.append(board.san(move) + note)
        board.push(move)
    parts.append(decoded.result)
    return " ".join(parts)

