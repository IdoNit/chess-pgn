"""Ask the AI helper for each test-set game's players/event/year, and show what survives the check
(a value is kept only if the heading prints it).

    uv run python -m spike.headings
"""

import json
import os

import bench.run as run
from chesspgn.pipeline import NoGameFound, _headers_read, convert
from chesspgn.repair import OllamaRowReader


def main() -> None:
    helper = OllamaRowReader(os.environ.get("OLLAMA_URL", "http://localhost:11434"), "qwen3-vl:8b-instruct")
    for manifest in sorted(run.BOOKS.glob("*/book.json")):
        info = json.loads(manifest.read_text(encoding="utf-8"))
        for entry in info["games"]:
            try:
                games = convert([manifest.parent / "pages" / p for p in entry["pages"]], run.SavedText())
            except NoGameFound:
                continue
            game = max(games, key=lambda g: g.move_count)
            read = _headers_read(game, helper)
            h = read.headers
            print(f"{manifest.parent.name:16} {entry['id']}  rules: {game.headers['White']} - {game.headers['Black']}"
                  f"  |  AI: {h['White']} - {h['Black']}, {h['Event']}, {h['Site']}, {h['Date']}  ->  {read.filename}",
                  flush=True)


if __name__ == "__main__":
    main()
