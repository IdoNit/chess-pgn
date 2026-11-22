import shutil
from pathlib import Path

import pytest

from chesspgn.pipeline import NoGameFound, convert
from chesspgn.reader import OllamaReader, ReaderUnavailable
from tests.conftest import expected_moves, fixture, fixture_path, read


class FakeReader:
    def __init__(self, texts: dict[str, str] | None = None, returns: str | None = None):
        self.texts = texts or {}
        self.returns = returns

    def read(self, image: Path) -> str:
        if self.returns is not None:
            return self.returns
        return self.texts[image.name]


def test_convert_stein():
    [g] = convert([fixture_path("stein.jpg")], FakeReader({"stein.jpg": fixture("stein.ocr.txt")}))
    assert g.filename == "Stein-Korchnoi - Moscow 1964.pgn"
    game = read(g.pgn)
    assert game.errors == []
    assert list(game.mainline_moves()) == expected_moves("stein.pgn")
    assert (game.headers["White"], game.headers["Result"], g.move_count, g.continues) == ("Stein", "1-0", 93, False)


def test_convert_figurine_page():
    [g] = convert([fixture_path("razov.jpg")], FakeReader({"razov.jpg": fixture("razov.ocr.txt")}))
    assert list(read(g.pgn).mainline_moves()) == expected_moves("razov.pgn")


def test_no_game_raises():  # Review Focus 1
    with pytest.raises(NoGameFound):
        convert([fixture_path("stein.jpg")], FakeReader(returns="Chapter 3. Some prose only."))


def test_split_across_pages(tmp_path):  # Review Focus 2
    text = fixture("stein.ocr.txt")
    cut = text.index("23...P-K4!")
    page1, page2 = tmp_path / "p1.jpg", tmp_path / "p2.jpg"
    shutil.copy(fixture_path("stein.jpg"), page1)
    shutil.copy(fixture_path("stein.jpg"), page2)
    [g] = convert([page1, page2], FakeReader({"p1.jpg": text[:cut], "p2.jpg": text[cut:]}))
    assert list(read(g.pgn).mainline_moves()) == expected_moves("stein.pgn")


def test_ollama_down():  # Review Focus 3
    with pytest.raises(ReaderUnavailable):
        OllamaReader("http://127.0.0.1:9").read(fixture_path("stein.jpg"))


@pytest.mark.live
def test_live_glm_ocr_razov():
    [g] = convert([fixture_path("razov.jpg")], OllamaReader("http://localhost:11434"))
    assert list(read(g.pgn).mainline_moves()) == expected_moves("razov.pgn")
