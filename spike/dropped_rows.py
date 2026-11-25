"""Ask the vision model about the rows the OCR dropped altogether (the app's real path and crop), and save
the answers next to the pages (pNNN.rows8.json) for the benchmark to replay.

Answers already saved (pNNN.vlm8.json for damaged rows, pNNN.rows8.json) are replayed, not asked again.

    uv run python -m spike.dropped_rows --split dev
"""

import argparse
import json
import os
import time
from pathlib import Path

import bench.run as run
from bench.vlm_answers import ROWS_SUFFIX, SavedRowReader, named_strip
from chesspgn import repair
from chesspgn.pipeline import NoGameFound, convert
from chesspgn.repair import OllamaRowReader


class Recording:
    def __init__(self, model: str) -> None:
        self.saved = SavedRowReader()
        self.live = OllamaRowReader(os.environ.get("OLLAMA_URL", "http://localhost:11434"), model)
        self.asked = 0

    def read_row(self, strip, number):
        rows = Path(strip.filename).with_suffix(ROWS_SUFFIX)
        known = json.loads(rows.read_text(encoding="utf-8")) if rows.exists() else {}
        if strip.info.get("dropped") and str(number) in known:
            return tuple(known[str(number)]) if known[str(number)] else None
        if not strip.info.get("dropped"):
            return self.saved.read_row(strip, number)  # a row the OCR has: answered in the earlier run
        began = time.monotonic()
        answer = self.live.read_row(strip, number)
        self.asked += 1
        print(f"  {Path(strip.filename).name} row {number:3}  {time.monotonic() - began:5.0f}s  {answer}", flush=True)
        known[str(number)] = list(answer) if answer else None
        rows.write_text(json.dumps(known, indent=1, ensure_ascii=False), encoding="utf-8")
        return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--model", default="qwen3-vl:8b-instruct")
    args = parser.parse_args()
    repair.row_strip = named_strip
    reader = Recording(args.model)
    for manifest in sorted(run.BOOKS.glob("*/book.json")):
        info = json.loads(manifest.read_text(encoding="utf-8"))
        if info["split"] != args.split:
            continue
        book = manifest.parent
        for entry in info["games"]:
            print(f"{book.name} {entry['id']}", flush=True)
            try:
                convert([book / "pages" / p for p in entry["pages"]], run.SavedText(), row_reader=reader)
            except NoGameFound:
                pass
    print(f"done: {reader.asked} new questions")


if __name__ == "__main__":
    main()
