"""Why each review click happens: the state of the page text for the move behind it.

    uv run python -m bench.causes --split dev --ai

    game boundary   "the game ends here" / a move added after the end (the game was cut wrong)
    missing row     nothing is printed for the move at its number (the OCR dropped it)
    damaged text    something is printed there but reads as no legal move
    misread         the printed text is one OCR correction away from the move
    note conflict   the move is printed exactly, but the same number also prints another legal move (a note)
    decoder doubt   the move is printed exactly and alone: the flag comes from the decoder's own caution
"""

import argparse
import json
from collections import Counter

import chess

import bench.run as run
from bench.clicks import review
from bench.extraction import numbered_entries
from bench.reason_answers import _best_cost
from bench.vlm_answers import SavedRowReader, whole_page
from chesspgn import repair
from chesspgn.decoder import COST_SUBSTITUTION
from chesspgn.pipeline import NoGameFound, convert


def cause(action: str, ply: int, expected: list[str], entries: dict) -> str:
    if action in ("end", "add") or ply > len(expected):
        return "game boundary"
    board = chess.Board()
    for san in expected[: ply - 1]:
        board.push_san(san)
    move = board.parse_san(expected[ply - 1])
    printed = entries.get((board.fullmove_number, board.turn == chess.BLACK), [])
    if not printed:
        return "missing row"
    own = _best_cost(board, move, printed)
    if own is None or own > COST_SUBSTITUTION:
        return "damaged text"
    if own == COST_SUBSTITUTION:
        return "misread"
    others = [m for m in board.legal_moves if m != move and (c := _best_cost(board, m, printed)) is not None
              and c <= COST_SUBSTITUTION]
    return "note conflict" if others else "decoder doubt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--ai", action="store_true")
    parser.add_argument("--detail", help="list the clicks of this cause")
    args = parser.parse_args()
    row_reader = None
    if args.ai:
        row_reader = SavedRowReader()
        repair.row_strip = whole_page
    totals: Counter = Counter()
    by_action: Counter = Counter()
    for manifest in sorted(run.BOOKS.glob("*/book.json")):
        info = json.loads(manifest.read_text(encoding="utf-8"))
        if info["split"] != args.split:
            continue
        book = manifest.parent
        for entry in info["games"]:
            expected = run.sans((book / "expected" / f"{entry['id']}.pgn").read_text(encoding="utf-8"))
            accept = {int(k): v for k, v in entry.get("accept", {}).items()}
            try:
                games = convert([book / "pages" / p for p in entry["pages"]], run.SavedText(), row_reader=row_reader)
            except NoGameFound:
                continue
            game, _ = run.best_match(expected, games, accept)
            entries = numbered_entries(game.text)
            log: list[tuple[str, int]] = []
            review(game, expected, accept, log=log)
            here = Counter(cause(a, p, expected, entries) for a, p in log)
            for a, p in log:
                by_action[(cause(a, p, expected, entries), a)] += 1
                if args.detail and cause(a, p, expected, entries) == args.detail:
                    side = "white" if p % 2 else "black"
                    print(f"    {entry['id']} ply {p} = move {(p + 1) // 2} {side}: book {expected[p - 1]} ({a})")
            totals.update(here)
            print(f"{book.name:16} {entry['id']}  {len(log):3} clicks  " + "  ".join(f"{k}: {v}" for k, v in here.most_common()),
                  flush=True)
    print("TOTAL", sum(totals.values()), "clicks")
    for name, n in totals.most_common():
        detail = ", ".join(f"{a} {c}" for (k, a), c in sorted(by_action.items()) if k == name)
        print(f"  {name:14} {n:3}   ({detail})")


if __name__ == "__main__":
    main()
