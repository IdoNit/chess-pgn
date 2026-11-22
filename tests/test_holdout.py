"""Held-out acceptance: run once on the finished build, with Ollama running (`pytest -m holdout -s`).

Pages in examples/holdout/ are never opened during development. Each image (or images sharing a stem prefix
"name-1.jpg", "name-2.jpg") must have a matching "name.pgn" with the correct game.
"""

import re
from pathlib import Path

import chess
import pytest

from chesspgn.pipeline import convert
from chesspgn.reader import OllamaReader
from tests.conftest import read

HOLDOUT = Path(__file__).parent.parent / "examples" / "holdout"


def holdout_cases() -> list[tuple[str, list[Path], Path]]:
    if not HOLDOUT.exists():
        return []
    cases = []
    for pgn in sorted(HOLDOUT.glob("*.pgn")):
        images = sorted(p for p in HOLDOUT.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                        and re.fullmatch(re.escape(pgn.stem) + r"(-\d+)?", p.stem))
        if images:
            cases.append((pgn.stem, images, pgn))
    return cases


@pytest.mark.holdout
@pytest.mark.parametrize("name,images,expected_pgn", holdout_cases(), ids=lambda v: v if isinstance(v, str) else "")
def test_holdout_page(name, images, expected_pgn):
    expected = list(read(expected_pgn.read_text(encoding="utf-8")).mainline_moves())
    games = convert(images, OllamaReader("http://localhost:11434"))
    game = max(games, key=lambda g: g.move_count)
    moves = list(read(game.pgn).mainline_moves())
    wrong = [i for i, (a, b) in enumerate(zip(moves, expected), start=1) if a != b]
    if len(moves) != len(expected):
        wrong.append(min(len(moves), len(expected)) + 1)
    flagged = sorted(f.ply for f in game.flags if not f.reason.startswith("side:"))
    print(f"\n{name}: {len(moves)}/{len(expected)} moves, wrong plies {wrong}, flags at {flagged}")
    unflagged = [w for w in wrong if not any(abs(p - w) <= 1 for p in flagged)]
    assert not unflagged, f"{name}: wrong moves with no flag within one ply: {unflagged}"
