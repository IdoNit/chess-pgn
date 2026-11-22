import random

import chess
import pytest

from chesspgn.decoder import decode_text
from tests.conftest import expected_moves, fixture
from tests.helpers import corrupt, render


def first_difference(a: list[chess.Move], b: list[chess.Move]) -> int | None:
    """1-based ply of the first difference, or None if equal."""
    for i, (x, y) in enumerate(zip(a, b), start=1):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b)) + 1


def test_stein_real_ocr_exact():
    d = decode_text(fixture("stein.ocr.txt"))
    assert d.convention == "descriptive"
    assert d.moves == expected_moves("stein.pgn")
    assert d.result == "1-0"


def test_stein_clean_has_no_flags():
    assert decode_text(fixture("stein.ocr.txt")).flags == []


def test_game_start_skips_table_and_prose():
    text = render(expected_moves("razov.pgn"), "english")
    table = "USSR Junior Ch, Tallinn 1948\n1 Korchnoi x 0 1 1 0 1 1 1 5\n2 Nei 1 x 0 = = 1 1 1 5\n"
    assert decode_text(table + text).moves == expected_moves("razov.pgn")


def test_result_from_and_won():
    assert decode_text(fixture("zikov.letters.txt")).result == "0-1"


def test_result_from_1_colon_0():
    assert decode_text(fixture("razov.letters.txt")).result == "1-0"


def test_qxp_capital_x_repaired_and_flagged():
    text = fixture("stein.ocr.txt").replace("47 QxP+", "47 QXP+")
    d = decode_text(text)
    assert d.moves == expected_moves("stein.pgn")
    assert any(f.ply == 93 and f.reason == "ocr" and f.chosen == "Qxf7+" for f in d.flags)


def test_icon_groups_learned():
    moves = expected_moves("razov.pgn")
    d = decode_text(render(moves, "groups"), convention="english")
    assert d.moves == moves
    assert d.mapping[:4] == (chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.ROOK)


def test_pins_override():
    moves = expected_moves("stein.pgn")
    pinned = chess.Move.from_uci("d3g6")  # 25. Bxg6 instead of the book's 25. Bxf6
    d = decode_text(fixture("stein.ocr.txt"), pins={49: pinned})
    assert d.moves[48] == pinned
    assert d.moves[:48] == moves[:48]


@pytest.mark.parametrize("n_errors,seed", [(3, 3), (6, 6), (15, 15)])
def test_damage_never_silent(n_errors, seed):
    rng = random.Random(seed)
    expected = expected_moves("stein.pgn")
    for _trial in range(4):
        damaged, log = corrupt(fixture("stein.ocr.txt"), n_errors, rng)
        d = decode_text(damaged, convention="descriptive")
        wrong_at = first_difference(d.moves, expected)
        if wrong_at is None:
            continue
        assert any(f.ply <= wrong_at + 1 for f in d.flags), f"silent error at ply {wrong_at}: {log}"


@pytest.mark.parametrize("style", ["english", "short_captures", "long", "german", "french", "russian", "figurine"])
def test_notation_styles_exact(style):
    moves = expected_moves("razov.pgn")
    assert decode_text(render(moves, style)).moves == moves
