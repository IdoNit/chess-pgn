"""Stress tests for the board decoder (spike).

  python3 stress.py notations            # every game rendered in many notations, convention auto-detected
  python3 stress.py corrupt <ocr.txt> <expected.pgn> <n_errors> <trials> <seed>
"""

import io
import random
import re
import sys
import time
from pathlib import Path

import chess
import chess.pgn

import decoder

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
GAMES = ["1. Korchnoi-Razov - Leningrad 1946.pgn", "2. Zikov-Korchnoi - Leningrad 1946.pgn", "Stein-Korchnoi_Moscow_1964.pgn"]
LETTERS = {
    "german": {"K": "K", "Q": "D", "R": "T", "B": "L", "N": "S"},
    "french": {"K": "R", "Q": "D", "R": "T", "B": "F", "N": "C"},
    "russian": {"K": "Кр", "Q": "Ф", "R": "Л", "B": "С", "N": "К"},
}
FIG = {"K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘"}


def mainline(name: str) -> list[chess.Move]:
    game = chess.pgn.read_game(io.StringIO((EXAMPLES / name).read_text(encoding="utf-8")))
    return list(game.mainline_moves())


def render(moves: list[chess.Move], style: str) -> str:
    board = chess.Board()
    out: list[str] = []
    for move in moves:
        if board.turn == chess.WHITE:
            out.append(f"{board.fullmove_number}")
        san = board.san(move)
        if style == "short_captures" and board.piece_type_at(move.from_square) == chess.PAWN and board.is_capture(move):
            san = chess.square_name(move.from_square)[0] + chess.square_name(move.to_square)[0]
            san += "+" if board.gives_check(move) else ""
        elif style == "long":
            san = board.lan(move)
        elif style in LETTERS:
            san = re.sub(r"[KQRBN]", lambda m: LETTERS[style][m.group(0)], san) if not san.startswith("O-") else san
            san = san.replace("O-O", "0-0")
        elif style in ("figurine", "figurine_stripped"):
            if not san.startswith("O-"):
                san = re.sub(r"[KQRBN]", lambda m: FIG[m.group(0)], san)
            if style == "figurine_stripped":
                san = re.sub(r"[♔♕♖♗♘]", "", san)
            # books abbreviate pawn captures too
            if board.piece_type_at(move.from_square) == chess.PAWN and board.is_capture(move):
                san = chess.square_name(move.from_square)[0] + chess.square_name(move.to_square)[0]
        out.append(san)
        board.push(move)
    return "Game 1: Some Header, Leningrad 1946\n" + " ".join(out)


def notations() -> None:
    styles = ["english", "short_captures", "long", "german", "french", "russian", "figurine", "figurine_stripped"]
    print(f"{'game':28} {'style':18} {'detected':12} {'result':22} {'repairs':>7} {'sec':>5}")
    for name in GAMES:
        moves = mainline(name)
        for style in styles:
            started = time.perf_counter()
            d = decoder.decode(render(moves, style))
            ok = d.moves == moves
            first_diff = next((i + 1 for i, (a, b) in enumerate(zip(d.moves, moves)) if a != b), None)
            verdict = "exact" if ok else f"{len(d.moves)}/{len(moves)} diff@{first_diff}"
            print(f"{name[:28]:28} {style:18} {d.convention:12} {verdict:22} {len(d.repairs):>7} {time.perf_counter()-started:5.1f}")


# ---------------------------------------------------------------- corruption

def corrupt(text: str, n_errors: int, rng: random.Random) -> tuple[str, list[str]]:
    """Apply OCR-style damage to tokens of the game text (not the comments)."""
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if re.search(r"[A-Z]-|x|^\d+$", w) and "(" not in w and ")" not in w]
    log: list[str] = []
    for i in rng.sample(candidates, min(n_errors, len(candidates))):
        w = words[i]
        kind = rng.choice(["confuse", "confuse", "drop_char", "drop_number", "garble"])
        if re.fullmatch(r"\d+", w.strip()):
            kind = "drop_number"
        if kind == "drop_number":
            new = "" if re.fullmatch(r"\d+", w.strip()) else w
        elif kind == "garble":
            new = "".join(rng.choice("#%&?~") for _ in w.strip()) if w.strip() else w
        elif kind == "drop_char":
            j = rng.randrange(len(w)) if w else 0
            new = w[:j] + w[j + 1:]
        else:
            positions = [j for j, ch in enumerate(w) if ch in decoder._SUBS]
            if not positions:
                continue
            j = rng.choice(positions)
            new = w[:j] + rng.choice(sorted(decoder._SUBS[w[j]])) + w[j + 1:]
        log.append(f"{kind}: {w!r}->{new!r}")
        words[i] = new
    return " ".join(words), log


def corruption(ocr_path: str, expected_path: str, n_errors: int, trials: int, seed: int) -> None:
    text = Path(ocr_path).read_text(encoding="utf-8")
    expected = list(chess.pgn.read_game(io.StringIO(Path(expected_path).read_text(encoding="utf-8"))).mainline_moves())
    rng = random.Random(seed)
    exact = flagged_wrong = silent_wrong = 0
    for trial in range(trials):
        damaged, log = corrupt(text, n_errors, rng)
        d = decoder.decode(damaged, "descriptive")
        if d.moves == expected:
            exact += 1
            continue
        first_diff = next((i + 1 for i, (a, b) in enumerate(zip(d.moves, expected)) if a != b), min(len(d.moves), len(expected)) + 1)
        flagged = any(r.ply <= first_diff + 1 for r in d.repairs) or (d.runner_up_differs_at is not None and d.runner_up_cost - d.cost < 1.5)
        if flagged:
            flagged_wrong += 1
        else:
            silent_wrong += 1
        print(f"  trial {trial}: {'FLAGGED' if flagged else 'SILENT'} wrong, {len(d.moves)}/{len(expected)} plies, first diff ply {first_diff}; damage: {log}")
    print(f"errors={n_errors}: exact {exact}/{trials}, wrong-but-flagged {flagged_wrong}, wrong-and-silent {silent_wrong}")


if __name__ == "__main__":
    if sys.argv[1] == "notations":
        notations()
    else:
        corruption(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
