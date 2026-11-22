import pytest

from chesspgn.icons import annotate, find_icons
from chesspgn.variations import build_game
from tests.conftest import expected_moves, fixture, fixture_path


@pytest.fixture(scope="module")
def razov_hits():
    return find_icons([fixture_path("razov.jpg")])


def test_razov_icons_found_and_grouped(razov_hits):
    assert len(razov_hits) >= 45
    assert 4 <= len({h.group for h in razov_hits}) <= 7


def test_hits_in_reading_order(razov_hits):
    assert [h.order for h in razov_hits] == sorted(h.order for h in razov_hits)


def test_no_icons_on_descriptive_page():
    assert find_icons([fixture_path("stein.jpg")]) == []


def test_no_hits_inside_diagram(razov_hits):
    # razov.jpg: the diagram occupies x 830-1280, y 1480-1940 (original pixels)
    assert not any(830 < h.box[0] < 1280 and 1480 < h.box[1] < 1940 for h in razov_hits)


def test_annotate_keeps_pawn_moves(razov_hits):
    text = annotate(fixture("razov.ocr.txt"), razov_hits)
    assert "4 d4 ed" in text
    assert "2 ①f3" in text or "2 ②f3" in text or any(f"2 {s}f3" in text for s in "①②③④⑤⑥⑦⑧")


def test_end_to_end_razov_main_exact(razov_hits):
    text = annotate(fixture("razov.ocr.txt"), razov_hits)
    assert list(build_game(text).game.mainline_moves()) == expected_moves("razov.pgn")


def test_end_to_end_zikov_main_exact():
    text = annotate(fixture("zikov.ocr.txt"), find_icons([fixture_path("zikov.jpg")]))
    assert list(build_game(text).game.mainline_moves()) == expected_moves("zikov.pgn")
