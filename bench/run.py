"""Benchmark: convert every book game in tests/fixtures/books and score it against the book.

Uses the OCR text saved by `bench.ocr_pages` (no Ollama needed). From the project folder:

    uv run python -m bench.run                 # all books
    uv run python -m bench.run --split dev     # only the development books
    uv run python -m bench.run --show 3        # also print the 3 worst games' silent errors in detail

The number that must be 0 is "silent": wrong moves with no flag within one ply.
Results are saved to bench/results/<time>.json.
"""

import argparse
import io
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn

from bench.extraction import extraction
from bench.vlm_answers import VlmAssisted
from bench.score import Score, score
from chesspgn.pipeline import ConvertedGame, NoGameFound, convert

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ROOT / "tests" / "fixtures" / "books"


class SavedText:
    """A reader that returns the OCR text saved beside each page image."""

    def __init__(self, suffix: str = ".ocr.txt") -> None:
        self.suffix = suffix

    def read(self, image: Path) -> str:
        saved = image.with_suffix(self.suffix)
        if not saved.exists():
            raise FileNotFoundError(f"{saved} is missing: run `python -m bench.ocr_pages` first")
        return saved.read_text(encoding="utf-8")


def sans(pgn: str) -> list[str]:
    game = chess.pgn.read_game(io.StringIO(pgn))
    board, out = game.board(), []
    for move in game.mainline_moves():
        out.append(board.san(move))
        board.push(move)
    return out


def best_match(expected: list[str], games: list[ConvertedGame], accept: dict[int, list[str]],
               ai_plies: set[int] = frozenset()) -> tuple[ConvertedGame, Score]:
    scored = []
    for game in games:
        flagged = sorted({f.ply for f in game.flags if not f.reason.startswith("side:")} | set(ai_plies))
        scored.append((game, score(expected, sans(game.pgn), flagged, accept)))
    return max(scored, key=lambda gs: gs[1].correct)


def run_game(book: Path, entry: dict, reader: SavedText) -> dict:
    expected = sans((book / "expected" / f"{entry['id']}.pgn").read_text(encoding="utf-8"))
    accept = {int(k): v for k, v in entry.get("accept", {}).items()}
    started = time.monotonic()
    try:
        games = convert([book / "pages" / p for p in entry["pages"]], reader)
    except NoGameFound:
        games = []
    seconds = time.monotonic() - started
    repaired = set(getattr(reader, "repaired", set()))  # before the page text is read again below
    page_text = "\n".join(reader.read(book / "pages" / p) for p in entry["pages"])
    read = extraction(page_text, expected)
    row = {"book": book.name, "game": entry["id"], "seconds": round(seconds, 1),
           "text_ok": read.exact + read.fixable, "text_damaged": read.damaged, "text_missing": read.missing,
           "text_bad_plies": read.bad_plies}
    if not games:
        return row | asdict(Score(expected=len(expected), correct=0, wrong=len(expected), extra=0, unflagged_wrong=0)) | {
            "flags": 0, "found": False}
    game, result = best_match(expected, games, accept, {2 * n - 1 + black for n, black in repaired})
    main_flags = [f for f in game.flags if not f.reason.startswith("side:")]
    output = sans(game.pgn)
    return row | asdict(result) | {
        "found": True,
        "flags": len(main_flags),
        "side_flags": len(game.flags) - len(main_flags),
        "silent_detail": [
            f"ply {p}: got {output[p - 1] if p <= len(output) else '-'}"
            f", book {expected[p - 1] if p <= len(expected) else '-'}" for p in result.silent_plies],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"])
    parser.add_argument("--show", type=int, default=0)
    parser.add_argument("--books", type=Path, default=BOOKS)
    parser.add_argument("--vlm", choices=["labels", "repair", "labels+repair", "tiebreak", "tiebreak+repair"], help="use the vision model's answers")
    parser.add_argument("--suffix", default=".ocr.txt", help="which saved text to use (e.g. another OCR's output)")
    args = parser.parse_args()
    reader = SavedText(args.suffix)
    if args.vlm:
        reader = VlmAssisted(labels="labels" in args.vlm or "tiebreak" in args.vlm, repair="repair" in args.vlm,
                             tiebreak="tiebreak" in args.vlm)
    rows = []
    for manifest in sorted(args.books.glob("*/book.json")):
        info = json.loads(manifest.read_text(encoding="utf-8"))
        if args.split and info["split"] != args.split:
            continue
        for entry in info["games"]:
            if hasattr(reader, "reset"):
                reader.reset()
            row = run_game(manifest.parent, entry, reader) | {"split": info["split"]}
            rows.append(row)
            print(f"{row['book']:16} {row['game']}  text {row['text_ok']:3}/{row['expected']:3} "
                  f"(damaged {row['text_damaged']:2}, missing {row['text_missing']:2})   game {row['correct']:3}/{row['expected']:3} right  "
                  f"silent {row['unflagged_wrong']:2}  flags {row['flags']:3}  {'EXACT' if row['exact'] else ''}")
    print()
    for split in ("dev", "test"):
        part = [r for r in rows if r["split"] == split]
        if not part:
            continue
        total = sum(r["expected"] for r in part)
        right = sum(r["correct"] for r in part)
        text_ok = sum(r["text_ok"] for r in part)
        print(f"{split:5}: TEXT  {text_ok}/{total} moves readable where they belong ({text_ok / total:.1%}), "
              f"damaged {sum(r['text_damaged'] for r in part)}, missing {sum(r['text_missing'] for r in part)}")
        print(f"{split:5}: GAME  {len(part)} games, {right}/{total} moves right ({right / total:.1%}), "
              f"{sum(r['exact'] for r in part)} games exact, SILENT {sum(r['unflagged_wrong'] for r in part)}, "
              f"flags {sum(r['flags'] for r in part)}, decode {sum(r['seconds'] for r in part):.0f}s")
    for r in sorted(rows, key=lambda r: -r["unflagged_wrong"])[: args.show]:
        print(f"\n{r['book']} {r['game']}:", *r.get("silent_detail", []), sep="\n  ")
    out = ROOT / "bench" / "results" / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
