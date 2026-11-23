"""Read every book page in tests/fixtures/books with GLM-OCR once and keep the text next to the image.

Needs the Ollama app running with glm-ocr. From the project folder:

    uv run python -m bench.ocr_pages            # all pages not read yet (safe to stop and re-run)
    uv run python -m bench.ocr_pages --redo     # read every page again

Each page is saved as pNNN.ocr.txt beside pNNN.jpg, so the benchmark runs without Ollama afterwards.
"""

import argparse
import time
from pathlib import Path

from chesspgn.reader import OllamaReader, PageIncomplete

BOOKS = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "books"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--ollama", default="http://localhost:11434")
    args = parser.parse_args()
    reader = OllamaReader(args.ollama)
    pages = sorted(BOOKS.glob("*/pages/*.jpg"))
    todo = [p for p in pages if args.redo or not p.with_suffix(".ocr.txt").exists()]
    print(f"{len(todo)} of {len(pages)} pages to read")
    started = time.monotonic()
    for n, page in enumerate(todo, start=1):
        t = time.monotonic()
        try:
            text = reader.read(page)
        except PageIncomplete as e:
            print(f"[{n}/{len(todo)}] {page.parent.parent.name}/{page.name}: CUT OFF ({e})")
            continue
        page.with_suffix(".ocr.txt").write_text(text, encoding="utf-8")
        took = time.monotonic() - t
        left = (time.monotonic() - started) / n * (len(todo) - n)
        print(f"[{n}/{len(todo)}] {page.parent.parent.name}/{page.name}: {took:.0f}s (about {left / 60:.0f} min left)")


if __name__ == "__main__":
    main()
