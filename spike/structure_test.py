"""THROWAWAY spike: does an LLM that only *labels* lines (game vs commentary) beat the layout rule?

    A: the app's pipeline (GLM-OCR -> figurines if needed -> layout rule -> board decoder)
    B: same, but a local LLM picks the commentary lines (by number) instead of the layout rule

Runs on all example pages (examples/ via spike/cases.json, plus examples/holdout/). Needs the Ollama app with
glm-ocr and the LLM pulled. Usage, from the project folder:

    uv run python spike/structure_test.py                       # LLM: qwen3-vl:8b-instruct
    uv run python spike/structure_test.py --llm qwen3:8b        # another local model (ollama pull it first)

Everything is saved in spike/results/structure-<time>/ : OCR text, the LLM's labelled text, both PGNs, report.md.
"""

import argparse
import difflib
import io
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn
import httpx

from chesspgn.games import split_games
from chesspgn.icons import annotate, find_icons
from chesspgn.layout import mark_commentary
from chesspgn.pipeline import uses_piece_letters
from chesspgn.reader import OllamaReader
from chesspgn.variations import build_game

ROOT = Path(__file__).resolve().parent.parent
OLLAMA = "http://localhost:11434"

LABEL_PROMPT = """\
Below is OCR text of a page from a chess book, one numbered line per line. It contains a chess game mixed with
commentary paragraphs, headings and page headers.

List the numbers of the lines that are NOT part of the game's own move sequence: commentary sentences (even when
they mention moves), analysis inside sentences, headings, page headers, diagram captions.
Lines that only contain the game's moves (with move numbers, symbols, brackets) or the result are NOT listed.

Answer with a JSON list of numbers only, e.g. [1, 2, 5, 6].

@@LINES@@
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="qwen3-vl:8b-instruct")
    args = parser.parse_args()
    out = ROOT / "spike" / "results" / f"structure-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    rows = []
    for name, images, expected_pgn in cases():
        print(f"[{name}] reading ...", flush=True)
        started = time.perf_counter()
        ocr = "\n".join(OllamaReader(OLLAMA).read(image) for image in images)
        ocr_s = time.perf_counter() - started
        (out / f"{name}.ocr.txt").write_text(ocr, encoding="utf-8")
        hits = find_icons(images)
        expected = expected_moves(expected_pgn)

        base = ocr if uses_piece_letters(ocr) else annotate(ocr, hits)
        a = run_decoder(mark_commentary(base))
        print(f"    A done", flush=True)
        started = time.perf_counter()
        labelled = label(base, args.llm)
        llm_s = time.perf_counter() - started
        (out / f"{name}.labelled.txt").write_text(labelled, encoding="utf-8")
        changed = text_changes(base, labelled)
        b = run_decoder(labelled)
        print(f"    B done ({llm_s:.0f}s LLM)", flush=True)

        for tag, result in (("A", a), ("B", b)):
            (out / f"{name}.{tag}.pgn").write_text(result["pgn"], encoding="utf-8")
            rows.append(row(name, tag, result, expected, changed if tag == "B" else 0, ocr_s, llm_s if tag == "B" else 0))

    report = (
        f"# Structure test (LLM {args.llm})\n\n"
        "Main = correct moves before the first wrong one / expected. Wrong = wrong plies; unflagged = wrong plies with "
        "no flag within one ply (must be 0). Changed must be 0 (B can only add braces).\n\n"
        "| Page | Pipeline | Main | Wrong | Unflagged | Flags | Changed | OCR s | LLM s |\n|---|---|---|---|---|---|---|---|---|\n"
        + "\n".join(rows) + "\n"
    )
    (out / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report + f"\nSaved in {out}")
    return 0


def cases() -> list[tuple[str, list[Path], Path]]:
    found = []
    for case in json.loads((ROOT / "spike" / "cases.json").read_text(encoding="utf-8")):
        name = case["name"].split(" (")[0]
        found.append((name, [ROOT / "examples" / i for i in case["images"]], ROOT / "examples" / case["expected"]))
    holdout = ROOT / "examples" / "holdout"
    if holdout.exists():
        for pgn in sorted(holdout.glob("*.pgn")):
            images = sorted(p for p in holdout.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                            and re.fullmatch(re.escape(pgn.stem) + r"(-\d+)?", p.stem))
            if images:
                found.append((pgn.stem, images, pgn))
    return found


def expected_moves(pgn: Path) -> list[chess.Move]:
    text = pgn.read_text(encoding="utf-8")
    head, _, body = text.partition("\n\n")
    body = re.sub(r"\n\s*\n", "\n", body)  # some files have empty lines inside the movetext
    return list(chess.pgn.read_game(io.StringIO(head + "\n\n" + body)).mainline_moves())


def label(text: str, model: str) -> str:
    """Ask the LLM which lines are commentary; wrap those lines in { } (no other change is possible)."""
    lines = text.splitlines()
    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(lines, start=1))
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": LABEL_PROMPT.replace("@@LINES@@", numbered)}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 8192},
    }
    response = httpx.post(f"{OLLAMA}/api/chat", json=payload, timeout=1800)
    response.raise_for_status()
    answer = response.json()["message"]["content"]
    found = re.search(r"\[[\d,\s]*\]", answer)
    commentary = set(json.loads(found.group(0))) if found else set()
    return "\n".join("{" + line + "}" if i in commentary and line.strip() else line for i, line in enumerate(lines, start=1))


def text_changes(original: str, labelled: str) -> int:
    """Characters the LLM changed apart from inserting braces (whitespace ignored)."""
    a = re.sub(r"\s+", "", original)
    b = re.sub(r"\s+", "", labelled.replace("{", "").replace("}", ""))
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal")


def run_decoder(text: str) -> dict:
    segments = split_games(text)
    if not segments:
        return {"moves": [], "flags": [], "pgn": ""}
    built = max((build_game(s) for s in segments), key=lambda b: len(b.decoded.moves))
    return {
        "moves": built.decoded.moves,
        "flags": [f.ply for f in built.flags if not f.reason.startswith("side:")],
        "pgn": str(built.game) + "\n",
    }


def row(name, tag, result, expected, changed, ocr_s, llm_s) -> str:
    moves, flags = result["moves"], result["flags"]
    wrong = [i for i, (x, y) in enumerate(zip(moves, expected), start=1) if x != y]
    if len(moves) != len(expected):
        wrong.append(min(len(moves), len(expected)) + 1)
    prefix = (wrong[0] - 1) if wrong else len(expected)
    unflagged = [w for w in wrong if not any(abs(p - w) <= 1 for p in flags)]
    return (f"| {name} | {tag} | {prefix}/{len(expected)}{' ✅' if not wrong else ''} | {len(wrong)} | "
            f"{len(unflagged)} | {len(flags)} | {changed} | {ocr_s:.0f} | {llm_s:.0f} |")


if __name__ == "__main__":
    sys.exit(main())
