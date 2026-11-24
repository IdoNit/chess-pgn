"""Read one printed move token as *clues* and test legal moves against them.

A token is never parsed into a move directly. It is read under a notation convention into clues (piece,
destination, capture, ...), and each legal move in the current position is checked against those clues.
"""

import re
from dataclasses import dataclass

import chess

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
# "A figurine of icon group k": the decoder learns which piece each group is.
GROUP_SYMBOLS = "①②③④⑤⑥⑦⑧"

PENALTY_UNMARKED_CHECK = 0.3  # books (and OCR) drop "+" fairly often
PENALTY_FALSE_CHECK = 1.0  # the book said "+" (or "mate") and the move doesn't deliver it
PENALTY_UNSTATED_PROMOTION = 0.5

# Descriptive file names, the same for both colours (the a-file is the queen's rook file).
_DESC_FILES = ["QR", "QN", "QB", "Q", "K", "KB", "KN", "KR"]
_DESC_PIECES = {
    "K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN,
}


@dataclass(frozen=True)
class Clues:
    """What a token says about a move. None means the token says nothing about that property."""

    castle: str | None = None  # "K" or "Q"
    piece: chess.PieceType | None = None  # None together with piece_class: "some piece of icon group k"
    piece_class: int | None = None
    piece_side: str | None = None  # descriptive "QR"/"KN": which of the two pieces
    from_file: int | None = None
    from_rank: int | None = None  # absolute 0-7
    from_rel_rank: int | None = None  # descriptive "N/2": rank counted from the mover's side, 1-8
    from_files: tuple[int, ...] | None = None  # descriptive "KPxP" / "N(Q4)-K2": the file the piece stands on
    to_file: int | None = None
    to_files: tuple[int, ...] | None = None  # descriptive "B4": QB or KB file
    to_rank: int | None = None  # absolute 0-7
    to_rel_rank: int | None = None  # descriptive rank, 1-8 from the mover's side
    to_rel_ranks: tuple[int, ...] | None = None  # "BxN(B3)": books count the captured piece's rank from either side
    capture: bool | None = None
    captured: chess.PieceType | None = None
    captured_files: tuple[int, ...] | None = None
    captured_side: str | None = None
    promotion: chess.PieceType | None = None
    check: bool = False
    mate: bool = False


_SUFFIX = re.compile(r"(?P<chk>[+#]{1,2}|\(?ch\)?|mate)$")
_CASTLE = re.compile(r"^(?:[0O]-[0O](?:-[0O])?|Castles(?:\(?[QK]R?\)?)?)$")
_ALG = re.compile(
    r"^(?P<ff>[a-h])?(?P<fr>[1-8])?(?P<sep>[-x:])?(?P<tf>[a-h])(?P<tr>[1-8])?"
    r"(?:=?\(?(?P<promo>[^\W\d_a-z]{1,2})\)?)?(?:e\.?p\.?)?$"
)
_DESC_SQUARE = r"(?:Q[RNB]?|K[RNB]?|[RNB])?[1-8]"
_DESC_MOVE = re.compile(
    r"^(?P<mside>[QK])?(?P<mfile>[RNB](?=P))?(?P<mpiece>[KQRBNP])"
    rf"(?:/(?P<mrank>[1-8])|@(?P<msquare>{_DESC_SQUARE}))?"
    r"(?:-(?P<dfile>Q[RNB]?|K[RNB]?|[RNB])(?P<drank>[1-8])"
    rf"|x(?P<cside>[QK])?(?P<cfile>[RNB])?(?P<cpiece>[KQRBNP])(?:/(?P<crank>[1-8])|@(?P<csquare>{_DESC_SQUARE}))?)"
    r"(?:=?\(?(?P<promo>[QRBN])\)?)?(?:e\.?p\.?)?$"
)


def read_clues(token: str, convention: str) -> Clues | None:
    """Clues in `token` under `convention`, or None if it isn't a move in that convention."""
    token = token.strip()
    suffix = _SUFFIX.search(token)
    body = token[: suffix.start()] if suffix else token
    chk = suffix.group("chk") if suffix else ""
    check = "+" in chk or "ch" in chk
    mate = "#" in chk or chk == "mate"
    if not body:
        return None
    if _CASTLE.match(body):
        long_castle = body.count("0") + body.count("O") == 3 or "Q" in body
        return Clues(castle="Q" if long_castle else "K", check=check, mate=mate)
    if convention == "descriptive":
        return _descriptive(body, check, mate)
    return _algebraic(body, PIECE_BY_LETTER[convention], check, mate)


def _algebraic(body: str, letters: dict[str, chess.PieceType], check: bool, mate: bool) -> Clues | None:
    piece_class: int | None = None
    if body[0] in GROUP_SYMBOLS:
        piece, piece_class, body = None, GROUP_SYMBOLS.index(body[0]), body[1:]
    else:
        piece, body = _leading_piece(body, letters)
    match = _ALG.match(body)
    if match is None:
        return None
    ff, fr, tf, tr, sep, promo_text = (match.group(k) for k in ("ff", "fr", "tf", "tr", "sep", "promo"))
    promo = None
    if promo_text is not None:
        if promo_text not in letters:
            return None
        promo = letters[promo_text]
    if tr is None and not (piece == chess.PAWN and ff is not None):
        return None  # only "ed"-style pawn captures may leave out the destination rank
    # "ed"/"exd5" (pawn, from-file, no from-rank) is a capture; "e2-e4" is not.
    short_pawn_capture = piece == chess.PAWN and ff is not None and fr is None
    return Clues(
        piece=piece,
        piece_class=piece_class,
        from_file=_file(ff),
        from_rank=_rank(fr),
        to_file=_file(tf),
        to_rank=_rank(tr),
        capture=True if sep in ("x", ":") or short_pawn_capture else None,
        promotion=promo,
        check=check,
        mate=mate,
    )


def _leading_piece(body: str, letters: dict[str, chess.PieceType]) -> tuple[chess.PieceType, str]:
    """The piece named by a leading figurine or letter (a pawn if none), and the rest of the token."""
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
    if g["mside"] and piece not in (chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.PAWN):
        return None
    from_files, from_rel_rank = None, int(g["mrank"]) if g["mrank"] else None
    if piece == chess.PAWN and (g["mside"] or g["mfile"]):  # "KPxP", "QBPxP": the pawn's file
        from_files = _desc_files(f"{g['mside'] or ''}{g['mfile'] or ''}")
    if g["msquare"]:  # "N@Q4-K2" (printed "Kt(Q4)-K2"): the square the piece leaves
        from_files, from_rel_rank = _desc_files(g["msquare"][:-1]) or None, int(g["msquare"][-1])
    common = {
        "piece": piece,
        "piece_side": g["mside"] if piece != chess.PAWN else None,
        "from_files": from_files,
        "from_rel_rank": from_rel_rank,
        "promotion": _DESC_PIECES[g["promo"]] if g["promo"] else None,
        "check": check,
        "mate": mate,
    }
    if g["drank"]:
        return Clues(to_files=_desc_files(g["dfile"]), to_rel_rank=int(g["drank"]), capture=False, **common)
    captured = _DESC_PIECES[g["cpiece"]]
    captured_files = None
    if captured == chess.PAWN and (g["cside"] or g["cfile"]):
        captured_files = _desc_files(f"{g['cside'] or ''}{g['cfile'] or ''}")
    captured_side = g["cside"] if captured in (chess.ROOK, chess.KNIGHT, chess.BISHOP) else None
    if g["csquare"]:  # "QxP@Q2" (printed "QxP (Q2)"): where the captured piece stands
        rank = int(g["csquare"][-1])
        return Clues(capture=True, captured=captured, captured_files=captured_files, captured_side=captured_side,
                     to_files=_desc_files(g["csquare"][:-1]) or None, to_rel_ranks=(rank, 9 - rank), **common)
    return Clues(
        capture=True,
        captured=captured,
        captured_files=captured_files,
        captured_side=captured_side,
        to_rel_rank=int(g["crank"]) if g["crank"] else None,
        **common,
    )


def _desc_files(name: str) -> tuple[int, ...]:
    if name in _DESC_FILES:
        return (_DESC_FILES.index(name),)
    return tuple(i for i, f in enumerate(_DESC_FILES) if f.endswith(name))  # "B" -> QB or KB


def _file(c: str | None) -> int | None:
    return None if c is None else ord(c) - ord("a")


def _rank(c: str | None) -> int | None:
    return None if c is None else int(c) - 1


# ---------------------------------------------------------------- matching


def match_cost(board: chess.Board, move: chess.Move, clues: Clues) -> float | None:
    """None if `move` contradicts the clues; otherwise a small penalty for soft mismatches."""
    if clues.castle is not None:
        if not board.is_castling(move):
            return None
        is_long = chess.square_file(move.to_square) < 4
        if is_long != (clues.castle == "Q"):
            return None
        return _check_penalty(board, move, clues)
    if board.is_castling(move):
        return None
    moving = board.piece_type_at(move.from_square)
    if clues.piece_class is not None:
        if moving == chess.PAWN:  # a figurine is never a pawn; which piece it is, the decoder checks
            return None
    elif moving != clues.piece:
        return None
    if not (_from_ok(board, move, clues) and _to_ok(board, move, clues) and _capture_ok(board, move, clues)):
        return None
    if clues.promotion is not None and move.promotion != clues.promotion:
        return None
    cost = PENALTY_UNSTATED_PROMOTION if move.promotion is not None and clues.promotion is None else 0.0
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
    if c.from_files is not None and chess.square_file(sq) not in c.from_files:
        return False
    if c.piece_side is not None and _which_side(board, sq) != c.piece_side:
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
    if c.to_rel_ranks is not None and _rel_rank(board, sq) not in c.to_rel_ranks:
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
    if board.piece_type_at(captured_sq) != c.captured:
        return False
    if c.captured_files is not None and chess.square_file(captured_sq) not in c.captured_files:
        return False
    if c.captured_side is not None and _which_side(board, captured_sq) != c.captured_side:
        return False
    return True


def _which_side(board: chess.Board, square: chess.Square) -> str:
    """Descriptive 'Q'/'K' identity of a rook, knight or bishop, judged from where it stands now."""
    if board.piece_type_at(square) == chess.BISHOP:
        # White's queen's bishop started on c1 (dark square), Black's on c8 (light square).
        dark = (chess.square_file(square) + chess.square_rank(square)) % 2 == 0
        return "Q" if dark == (board.color_at(square) == chess.WHITE) else "K"
    return "Q" if chess.square_file(square) < 4 else "K"


def _check_penalty(board: chess.Board, move: chess.Move, c: Clues) -> float:
    if c.mate:
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        return 0.0 if is_mate else PENALTY_FALSE_CHECK
    gives_check = board.gives_check(move)
    if c.check and not gives_check:
        return PENALTY_FALSE_CHECK
    if gives_check and not c.check:
        return PENALTY_UNMARKED_CHECK
    return 0.0
