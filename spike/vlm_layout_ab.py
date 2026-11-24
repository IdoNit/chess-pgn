"""EXPERIMENT: which way of asking "move row or paragraph?" does a small vision model answer reliably?

Runs a fixed sample of numbered lines from the Capablanca book (spike/layout_sample.json, each with the right
answer) through a few ways of asking, and prints how often each is right. From the project folder:

    uv run python -m spike.vlm_layout_ab
    uv run python -m spike.vlm_layout_ab --model qwen3-vl:8b-instruct
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from spike.vlm_rows import BOOKS, Q1, ask, boxed, neighbourhood

SAMPLE = Path(__file__).resolve().parent / "layout_sample.json"

CONTRAST = """Chess books print the moves of the game as short separate lines: a move number with one or two moves,
often lined up in columns. Notes are ordinary paragraphs of sentences, which may mention moves.
Is the line in the red box one of the short move lines (A), or part of a paragraph (B)? Answer A or B only."""


VARIANTS = {
    "layout question, close-up": lambda page, box: (Q1, neighbourhood(page, box)),
    "contrast question, close-up": lambda page, box: (CONTRAST, neighbourhood(page, box)),
    "contrast question, whole page": lambda page, box: (CONTRAST, boxed(page, box)),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-vl:4b-instruct")
    args = parser.parse_args()
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    print(f"{len(sample)} lines ({sum(s['truth'] == 'A' for s in sample)} move rows, "
          f"{sum(s['truth'] == 'B' for s in sample)} note lines), model {args.model}\n")
    for name, build in VARIANTS.items():
        right = {"A": 0, "B": 0}
        seconds = 0.0
        for item in sample:
            page = Image.open(BOOKS / item["page"].replace("/", "/pages/"))
            prompt, image = build(page, tuple(item["box"]))
            answer, took = ask(args.model, prompt, image, as_json=False)
            seconds += took
            right[item["truth"]] += answer.strip().upper()[:1] == item["truth"]
        rows = sum(s["truth"] == "A" for s in sample)
        notes = len(sample) - rows
        print(f"{name:32} move rows {right['A']}/{rows}   note lines {right['B']}/{notes}   "
              f"{seconds / len(sample):.1f}s each")


if __name__ == "__main__":
    main()
