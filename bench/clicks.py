"""Clicks to a correct game: simulate the review screen until the game matches the book.

The reviewer walks the game in order. At the first wrong move they fix it the way the review screen allows,
the game is decoded again with that fix pinned (as the app does), and they carry on from there.

    confirm  a flagged move that is right: one click on "confirm"
    pick     a wrong flagged move whose book move is one of the offered buttons: one click
    drag     a wrong move fixed by dragging the piece on the board (book move not offered)
    hunt     a wrong move with no flag within one ply: the reviewer must find it themselves (then drags it)
    reshown  confirmed flags the app shows again after a later fix (the app forgets confirmations on re-decode)
    left     the game still differs from the book when the simulation gives up (fix refused or no progress)

    uv run python -m bench.clicks --split dev --ai
"""

import argparse
import io
import json
from pathlib import Path

import chess
import chess.pgn

import bench.run as run
from bench.vlm_answers import SavedRowReader, whole_page
from chesspgn import repair
from chesspgn.decoder import suggest_moves
from chesspgn.pipeline import ConvertedGame, NoGameFound, convert, game_from_text

MAX_FIXES = 200


def _moves(pgn: str) -> list[chess.Move]:
    return list(chess.pgn.read_game(io.StringIO(pgn)).mainline_moves())


def _first_wrong(played: list[chess.Move], expected: list[str], accept: dict[int, list[str]]) -> int | None:
    """1-based ply of the first move that differs from the book (or is missing / extra), None if equal."""
    board = chess.Board()
    for ply, san in enumerate(expected, start=1):
        if ply > len(played):
            return ply
        got = board.san(played[ply - 1])
        if got != san and got not in accept.get(ply, []):
            return ply
        board.push(played[ply - 1])
    return len(expected) + 1 if len(played) > len(expected) else None


def review(game: ConvertedGame, expected: list[str], accept: dict[int, list[str]]) -> dict:
    counts = dict(confirm=0, pick=0, drag=0, hunt=0, reshown=0, left=0, fixes=0)
    pins: dict[int, chess.Move] = {}
    confirmed: dict[int, str] = {}  # ply -> uci the reviewer confirmed there
    last_wrong = 0
    while True:
        played = _moves(game.pgn)
        flags = {f.ply: f for f in game.flags if not f.reason.startswith("side:") and f.ply not in pins}
        wrong = _first_wrong(played, expected, accept)
        end = (wrong - 1) if wrong is not None else 10**6
        for ply, flag in sorted(flags.items()):
            if ply > end:
                break
            uci = played[ply - 1].uci() if ply <= len(played) else flag.reason
            if confirmed.get(ply) == uci:
                counts["reshown"] += 1
            else:
                confirmed[ply] = uci
                counts["confirm"] += 1
        if wrong is None:
            return counts
        if wrong > len(expected):  # the game runs on past the book's last move: no way to cut it in the app
            counts["left"] = len(played) - len(expected)
            return counts
        if wrong <= last_wrong and wrong in pins or counts["fixes"] >= MAX_FIXES:
            counts["left"] = len(expected) - wrong + 1
            return counts
        board = chess.Board()
        for move in played[: wrong - 1]:
            board.push(move)
        target = board.parse_san(expected[wrong - 1])
        flag = flags.get(wrong)
        if wrong <= len(played) and flag is not None:
            offered = [flag.chosen, *suggest_moves(board, flag.token, played[wrong - 1])]
            counts["pick" if board.san(target) in offered else "drag"] += 1
        elif any(p in flags for p in (wrong - 1, wrong + 1)):
            counts["drag"] += 1
        else:
            counts["hunt"] += 1
        counts["fixes"] += 1
        pins[wrong] = target
        last_wrong = wrong
        game = game_from_text(game.text, dict(pins), game.headers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"])
    parser.add_argument("--ai", action="store_true")
    args = parser.parse_args()
    row_reader = None
    if args.ai:
        row_reader = SavedRowReader()
        repair.row_strip = whole_page
    reader = run.SavedText()
    totals: dict[str, int] = {}
    for manifest in sorted(run.BOOKS.glob("*/book.json")):
        info = json.loads(manifest.read_text(encoding="utf-8"))
        if args.split and info["split"] != args.split:
            continue
        book = manifest.parent
        for entry in info["games"]:
            expected = run.sans((book / "expected" / f"{entry['id']}.pgn").read_text(encoding="utf-8"))
            accept = {int(k): v for k, v in entry.get("accept", {}).items()}
            try:
                games = convert([book / "pages" / p for p in entry["pages"]], reader, row_reader=row_reader)
            except NoGameFound:
                print(f"{book.name:16} {entry['id']}  no game found")
                continue
            game, _score = run.best_match(expected, games, accept)
            counts = review(game, expected, accept)
            clicks = counts["confirm"] + counts["pick"] + counts["drag"] + counts["hunt"]
            print(f"{book.name:16} {entry['id']}  {len(expected):3} plies  clicks {clicks:3}  "
                  + "  ".join(f"{k} {v:2}" for k, v in counts.items()), flush=True)
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v
            totals["clicks"] = totals.get("clicks", 0) + clicks
            totals["plies"] = totals.get("plies", 0) + len(expected)
            totals["games"] = totals.get("games", 0) + 1
    print("TOTAL", "  ".join(f"{k} {v}" for k, v in totals.items()))


if __name__ == "__main__":
    main()
