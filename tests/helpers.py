"""Test-only helpers: render a game in other notations, and damage text the way OCR does."""

import random
import re

import chess

from chesspgn.tokens import _SUBSTITUTES

LETTERS = {
    "german": {"K": "K", "Q": "D", "R": "T", "B": "L", "N": "S"},
    "french": {"K": "R", "Q": "D", "R": "T", "B": "F", "N": "C"},
    "russian": {"K": "Кр", "Q": "Ф", "R": "Л", "B": "С", "N": "К"},
}
FIGURINE = {"K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘"}
GROUPS = {"N": "①", "B": "②", "Q": "③", "R": "④", "K": "⑤"}


def render(moves: list[chess.Move], style: str) -> str:
    board = chess.Board()
    out: list[str] = []
    for move in moves:
        if board.turn == chess.WHITE:
            out.append(str(board.fullmove_number))
        out.append(_token(board, move, style))
        board.push(move)
    return "Game 1: Some Header, Leningrad 1946\n" + " ".join(out)


def _token(board: chess.Board, move: chess.Move, style: str) -> str:
    san = board.san(move)
    pawn_capture = board.piece_type_at(move.from_square) == chess.PAWN and board.is_capture(move)
    short = chess.square_name(move.from_square)[0] + chess.square_name(move.to_square)[0]
    check = "+" if board.gives_check(move) else ""
    if style == "long":
        return board.lan(move)
    if san.startswith("O-"):
        return san.replace("O", "0") if style != "english" else san
    if style in ("short_captures", "figurine", "groups") and pawn_capture:
        return short + check
    if style in LETTERS:
        return re.sub(r"[KQRBN]", lambda m: LETTERS[style][m.group(0)], san)
    if style == "figurine":
        return re.sub(r"^[KQRBN]", lambda m: FIGURINE[m.group(0)], san)
    if style == "groups":
        return re.sub(r"^[KQRBN]", lambda m: GROUPS[m.group(0)], san)
    return san


def corrupt(text: str, n_errors: int, rng: random.Random) -> tuple[str, list[str]]:
    """Apply OCR-style damage to move tokens and move numbers."""
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if re.search(r"[A-Z]-|x|^\d+$", w) and not re.search(r"[()]", w)]
    log: list[str] = []
    for i in rng.sample(candidates, min(n_errors, len(candidates))):
        w = words[i]
        kind = "drop_number" if re.fullmatch(r"\d+", w.strip()) else rng.choice(["confuse", "confuse", "drop_char", "garble"])
        if kind == "drop_number":
            new = ""
        elif kind == "garble":
            new = "".join(rng.choice("#%&?~") for _ in w.strip())
        elif kind == "drop_char":
            j = rng.randrange(len(w))
            new = w[:j] + w[j + 1:]
        else:
            positions = [j for j, ch in enumerate(w) if ch in _SUBSTITUTES]
            if not positions:
                continue
            j = rng.choice(positions)
            new = w[:j] + rng.choice(sorted(_SUBSTITUTES[w[j]])) + w[j + 1:]
        log.append(f"{kind}: {w!r}->{new!r}")
        words[i] = new
    return " ".join(words), log
