"""Turn one printed move token into a set of matching legal moves.

A token is never "parsed into a move". Instead it is read as *clues* (piece, destination, capture, ...)
under a notation convention, and every legal move in the current position is tested against those clues.
"""

import re
from dataclasses import dataclass

import chess

# ---------------------------------------------------------------- conventions

PIECE_BY_LETTER: dict[str, dict[str, chess.PieceType]] = {
    "english": {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT},
    "german": {"K": chess.KING, "D": chess.QUEEN, "T": chess.ROOK, "L": chess.BISHOP, "S": chess.KNIGHT},
    "french": {"R": chess.KING, "D": chess.QUEEN, "T": chess.ROOK, "F": chess.BISHOP, "C": chess.KNIGHT},
    "spanish": {"R": chess.KING, "D": chess.QUEEN, "T": chess.ROOK, "A": chess.BISHOP, "C": chess.KNIGHT},
    "russian": {"Кр": chess.KING, "Ф": chess.QUEEN, "Л": chess.ROOK, "С": chess.BISHOP, "К": chess.KNIGHT},
}
FIGURINES: dict[str, chess.PieceType] = {
    "♔": chess.KING, "♚": chess.KING, "♕": chess.QUEEN, "♛": chess.QUEEN, "♖": chess.ROOK, "♜": chess.ROOK,
    "♗": chess.BISHOP, "♝": chess.BISHOP, "♘": chess.KNIGHT, "♞": chess.KNIGHT, "♙": chess.PAWN, "♟": chess.PAWN,
}
CONVENTIONS: list[str] = ["descriptive", *PIECE_BY_LETTER]
# Placeholders for "figurine of shape group k": the decoder learns which piece each group is.
CLASS_SYMBOLS = "①②③④⑤⑥⑦⑧"

# Descriptive file names, identical for both colours (a-file = queen's rook file).
_DESC_FILES = ["QR", "QN", "QB", "Q", "K", "KB", "KN", "KR"]
_DESC_PIECES = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN}

# ---------------------------------------------------------------- clues


@dataclass(frozen=True)
class Clues:
    """What a token says about a move. None = the token says nothing about it."""

    castle: str | None = None  # "K" or "Q"
    piece: chess.PieceType | None = None  # None with piece_class set = "some piece of icon group k"
    piece_class: int | None = None
    piece_side: str | None = None  # descriptive "QR"/"KN": which of the two pieces
    from_file: int | None = None
    from_rank: int | None = None  # absolute 0-7
    from_rel_rank: int | None = None  # descriptive "N/2": rank counted from the mover's side, 1-8
    to_file: int | None = None
    to_files: tuple[int, ...] | None = None  # descriptive "B4": QB or KB
    to_rank: int | None = None  # absolute 0-7
    to_rel_rank: int | None = None  # descriptive rank, 1-8 from mover's side
    capture: bool | None = None
    captured: chess.PieceType | None = None
    captured_files: tuple[int, ...] | None = None
    captured_side: str | None = None
    promotion: chess.PieceType | None = None
    check: bool = False
    mate: bool = False


_SUFFIX = re.compile(r"(?P<chk>[+#]{0,2}|\(?ch\)?|mate)$")
_CASTLE = re.compile(r"^(?:[0O]-[0O](?:-[0O])?|Castles(?:\(?[QK]R?\)?)?)$")
_ALG = re.compile(
    r"^(?P<piece>[A-Z])?(?P<ff>[a-h])?(?P<fr>[1-8])?(?P<sep>[-x:])?(?P<tf>[a-h])(?P<tr>[1-8])?"
    r"(?:=?\(?(?P<promo>[A-Z])\)?)?(?:e\.?p\.?)?$"
)
_DESC_MOVE = re.compile(
    r"^(?P<mside>[QK])?(?P<mpiece>[KQRBNP])(?:/(?P<mrank>[1-8]))?"
    r"(?:-(?P<dfile>Q[RNB]?|K[RNB]?|[RNB])(?P<drank>[1-8])"
    r"|x(?P<cside>[QK])?(?P<cfile>[RNB])?(?P<cpiece>[KQRBNP])(?:/(?P<crank>[1-8]))?)"
    r"(?:=?\(?(?P<promo>[QRBN])\)?)?(?:e\.?p\.?)?$"
)


def read_clues(token: str, convention: str) -> Clues | None:
    """Clues in `token` under `convention`, or None if the token isn't a move in that convention."""
    token = token.strip()
    suffix = _SUFFIX.search(token)
    chk = suffix.group("chk") if suffix else ""
    body = token[: suffix.start()] if suffix and chk else token
    check = "+" in chk or "ch" in chk
    mate = "#" in chk or chk == "mate"

    if _CASTLE.match(body):
        long_castle = body.count("0") + body.count("O") == 3 or "Q" in body
        return Clues(castle="Q" if long_castle else "K", check=check, mate=mate)
    if convention == "descriptive":
        return _descriptive(body, check, mate)
    return _algebraic(body, PIECE_BY_LETTER[convention], check, mate)


def _algebraic(body: str, letters: dict[str, chess.PieceType], check: bool, mate: bool) -> Clues | None:
    piece: chess.PieceType | None = chess.PAWN
    piece_class: int | None = None
    if body and body[0] in CLASS_SYMBOLS:
        piece, piece_class, body = None, CLASS_SYMBOLS.index(body[0]), body[1:]
    else:
        piece, body = _leading_piece(body, letters)
    match = _ALG.match(body)
    if match is None or match.group("piece") is not None:
        return None
    promo = None
    if match.group("promo"):
        promo = letters.get(match.group("promo"))
        if promo is None:
            return None
    ff, fr, tf, tr, sep = (match.group(k) for k in ("ff", "fr", "tf", "tr", "sep"))
    if tr is None and not (piece == chess.PAWN and ff is not None):
        return None  # only "ed"-style pawn captures may omit the destination rank
    return Clues(
        piece=piece,
        piece_class=piece_class,
        from_file=_file(ff),
        from_rank=_rank(fr),
        to_file=_file(tf),
        to_rank=_rank(tr),
        # "ed"/"exd5" (pawn, from-file, no from-rank) is a capture; "e2-e4" is not.
        capture=True if (sep in ("x", ":") or (piece == chess.PAWN and ff is not None and fr is None)) else None,
        promotion=promo,
        check=check,
        mate=mate,
    )


def _leading_piece(body: str, letters: dict[str, chess.PieceType]) -> tuple[chess.PieceType, str]:
    """Piece named by a leading figurine or letter (pawn if none), and the rest of the token."""
    for symbol, kind in FIGURINES.items():
        if body.startswith(symbol):
            return kind, body[len(symbol):]
    for letter in sorted(letters, key=len, reverse=True):  # "Кр" before "К"
        if body.startswith(letter):
            return letters[letter], body[len(letter):]
    return chess.PAWN, body


def _descriptive(body: str, check: bool, mate: bool) -> Clues | None:
    match = _DESC_MOVE.match(body)
    if match is None:
        return None
    g = match.groupdict()
    piece = _DESC_PIECES[g["mpiece"]]
    side = g["mside"] if piece in (chess.ROOK, chess.KNIGHT, chess.BISHOP) else None
    if g["mside"] and side is None and piece != chess.PAWN:
        return None
    common = {
        "piece": piece,
        "piece_side": side,
        "from_rel_rank": int(g["mrank"]) if g["mrank"] else None,
        "promotion": _DESC_PIECES[g["promo"]] if g["promo"] else None,
        "check": check,
        "mate": mate,
    }
    if g["drank"]:
        return Clues(to_files=_desc_files(g["dfile"]), to_rel_rank=int(g["drank"]), capture=False, **common)
    captured_files = _desc_files(f"{g['cside'] or ''}{g['cfile'] or ''}") if g["cfile"] or (
        g["cside"] and g["cpiece"] == "P"
    ) else None
    captured_side = g["cside"] if g["cpiece"] in "RNB" and g["cside"] else None
    if g["cpiece"] != "P":
        captured_files = None
    return Clues(
        capture=True,
        captured=_DESC_PIECES[g["cpiece"]],
        captured_files=captured_files,
        captured_side=captured_side,
        to_rel_rank=int(g["crank"]) if g["crank"] else None,
        **common,
    )


def _desc_files(name: str) -> tuple[int, ...]:
    if name in _DESC_FILES:
        return (_DESC_FILES.index(name),)
    return tuple(i for i, f in enumerate(_DESC_FILES) if f.endswith(name))  # "B" -> QB, KB


def _file(c: str | None) -> int | None:
    return None if c is None else ord(c) - ord("a")


def _rank(c: str | None) -> int | None:
    return None if c is None else int(c) - 1


# ---------------------------------------------------------------- matching


def match_cost(board: chess.Board, move: chess.Move, clues: Clues) -> float | None:
    """None if the move contradicts the clues; otherwise a small penalty for soft mismatches."""
    if clues.castle is not None:
        if not board.is_castling(move):
            return None
        is_long = chess.square_file(move.to_square) < 4
        return None if is_long != (clues.castle == "Q") else _check_penalty(board, move, clues)

    if board.is_castling(move):
        return None
    moving = board.piece_type_at(move.from_square)
    if clues.piece_class is not None:
        if moving == chess.PAWN:  # a figurine is never a pawn; which piece it is, the decoder checks
            return None
    elif moving != clues.piece:
        return None
    if not _from_ok(board, move, clues) or not _to_ok(board, move, clues):
        return None
    if not _capture_ok(board, move, clues):
        return None
    if clues.promotion is not None and move.promotion != clues.promotion:
        return None
    cost = 0.0
    if move.promotion is not None and clues.promotion is None:
        cost += 0.5
    return cost + _check_penalty(board, move, clues)


def _rel_rank(board: chess.Board, square: chess.Square) -> int:
    rank = chess.square_rank(square)
    return rank + 1 if board.turn == chess.WHITE else 8 - rank


def _from_ok(board: chess.Board, move: chess.Move, c: Clues) -> bool:
    sq = move.from_square
    if c.from_file is not None and chess.square_file(sq) != c.from_file:
        return False
    if c.from_rank is not None and chess.square_rank(sq) != c.from_rank:
        return False
    if c.from_rel_rank is not None and _rel_rank(board, sq) != c.from_rel_rank:
        return False
    if c.piece_side is not None and _which_side(board, sq, c.piece) != c.piece_side:
        return False
    return True


def _to_ok(board: chess.Board, move: chess.Move, c: Clues) -> bool:
    sq = move.to_square
    if c.to_file is not None and chess.square_file(sq) != c.to_file:
        return False
    if c.to_files is not None and chess.square_file(sq) not in c.to_files:
        return False
    if c.to_rank is not None and chess.square_rank(sq) != c.to_rank:
        return False
    if c.to_rel_rank is not None and _rel_rank(board, sq) != c.to_rel_rank:
        return False
    return True


def _capture_ok(board: chess.Board, move: chess.Move, c: Clues) -> bool:
    is_capture = board.is_capture(move)
    if c.capture is True and not is_capture:
        return False
    if c.capture is False and is_capture:
        return False
    if c.captured is None:
        return True
    if board.is_en_passant(move):
        captured_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
    else:
        captured_sq = move.to_square
    captured_type = board.piece_type_at(captured_sq)
    if captured_type != c.captured:
        return False
    if c.captured_files is not None and chess.square_file(captured_sq) not in c.captured_files:
        return False
    if c.captured_side is not None and _which_side(board, captured_sq, captured_type) != c.captured_side:
        return False
    return True


def _which_side(board: chess.Board, square: chess.Square, piece: chess.PieceType | None) -> str:
    """Descriptive 'Q'/'K' piece identity, approximated from where the piece stands now."""
    if piece == chess.BISHOP:
        # White's QB started on c1 (dark), Black's QB on c8 (light).
        dark = (chess.square_file(square) + chess.square_rank(square)) % 2 == 0
        color = board.color_at(square)
        return "Q" if dark == (color == chess.WHITE) else "K"
    return "Q" if chess.square_file(square) < 4 else "K"


def _check_penalty(board: chess.Board, move: chess.Move, c: Clues) -> float:
    gives_check = board.gives_check(move)
    if c.mate:
        board.push(move)
        mate = board.is_checkmate()
        board.pop()
        return 0.0 if mate else 1.0
    if c.check and not gives_check:
        return 1.0  # the book said "+": a non-checking move is very suspicious
    if gives_check and not c.check:
        return 0.3  # books (and OCR) sometimes drop "+"
    return 0.0
