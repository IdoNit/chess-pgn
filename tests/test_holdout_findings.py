"""Regression tests for what the held-out Marshall pages exposed (they are development examples now)."""

import json
from pathlib import Path

import chess
import httpx
import pytest

from chesspgn.layout import mark_commentary
from chesspgn.notation import GROUP_SYMBOLS
from chesspgn.pipeline import convert
from chesspgn.reader import OllamaReader, PageIncomplete
from chesspgn.tokens import Token, tokenize
from chesspgn.variations import build_game
from tests.conftest import expected_moves, fixture, fixture_path


class FakeReader:
    def __init__(self, texts: dict[str, str]):
        self.texts = texts

    def read(self, image: Path) -> str:
        return self.texts[image.name]


def assert_exact_or_flagged(moves: list[chess.Move], expected: list[chess.Move], flags) -> None:
    wrong = [i for i, (a, b) in enumerate(zip(moves, expected), start=1) if a != b]
    if len(moves) != len(expected):
        wrong.append(min(len(moves), len(expected)) + 1)
    plies = [f.ply for f in flags if not f.reason.startswith("side:")]
    unflagged = [w for w in wrong if not any(abs(p - w) <= 1 for p in plies)]
    assert not unflagged, f"wrong plies without a flag: {unflagged} (wrong: {wrong})"
    assert len(wrong) <= 2, f"too many wrong moves: {wrong}"


# --- the reader must not cut pages short ---------------------------------------------------------

def reader_with(handler) -> OllamaReader:
    return OllamaReader("http://ollama.test", transport=httpx.MockTransport(handler))


def test_reader_asks_for_room_to_read_a_whole_page():
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "1 e4 e5"}, "done_reason": "stop"})

    reader_with(handler).read(fixture_path("stein.jpg"))
    assert sent["options"]["num_ctx"] >= 16384 and sent["options"]["num_predict"] >= 8192


def test_reader_refuses_a_cut_off_page():
    def handler(request):
        return httpx.Response(200, json={"message": {"content": "1 e4 e5 2 Nf3"}, "done_reason": "length"})

    with pytest.raises(PageIncomplete):
        reader_with(handler).read(fixture_path("stein.jpg"))


# --- OCR'd move numbers ----------------------------------------------------------------------------

def test_ocr_numbers_with_letter_l():
    assert tokenize("ll B×d5? l3 Q×e7")[0] == Token("", number=11)
    assert tokenize("ll B×d5? l3 Q×e7")[2] == Token("", number=13)
    assert tokenize("I count")[0].text == "I"  # a lone "I" is a word, not a number


# --- letters are not figurines ---------------------------------------------------------------------

def test_letter_notation_page_gets_no_icon_groups():
    [game] = convert([fixture_path("fox.jpg")], FakeReader({"fox.jpg": fixture("fox.ocr.txt")}))
    assert not any(symbol in game.text for symbol in GROUP_SYMBOLS)


# --- commentary paragraphs are not the game ------------------------------------------------------

def test_commentary_paragraph_wrapped():
    marked = mark_commentary(fixture("fox.ocr.txt"))
    assert "{With 6 Nf3 followed by castling and Rel, White gets an edge.}" in marked
    assert "{After 12 Ba3}" in marked
    assert "4 e×d5 N×d5 5 Bg2 Be6 6 Nge2?!" in marked and "{4 e×d5" not in marked


def test_paragraph_continuation_starting_with_a_number_is_commentary():
    marked = mark_commentary(fixture("mason.ocr.txt"))
    assert "{28 Kg3! (28 ... Re3+ 29 Rf3 Rel 30 Rfl R×fl! 31 B×fl Ke7) and which he}" in marked
    assert "{38 Bg6 h4+ and now 39 Kg4 h3 or 39 K×h4 B×f4 wins.}" in marked
    assert "{or c3.}" in marked


def test_game_lines_and_results_kept():
    marked = mark_commentary(fixture("mason.ocr.txt"))
    assert "\n17 Be2 Rae8 18 f4 b6!" in marked and "\nRe3+ 35 Kf2" in marked and "\nresigns" in marked


def test_bracketed_commentary_pages_unchanged():
    for name in ("stein.ocr.txt", "razov.letters.txt", "zikov.letters.txt"):
        game = mark_commentary(fixture(name))
        assert list(build_game(game).game.mainline_moves()) == expected_moves(name.split(".")[0] + ".pgn")


@pytest.mark.parametrize("name", ["fox", "mason"])
def test_paragraph_commentary_books_decode(name):
    built = build_game(fixture(f"{name}.ocr.txt"))
    assert_exact_or_flagged(list(built.game.mainline_moves()), expected_moves(f"{name}.pgn"), built.flags)


def test_two_page_game_decodes():
    images = [fixture_path("steinitz-1.jpg"), fixture_path("steinitz-2.jpg")]
    texts = {"steinitz-1.jpg": fixture("steinitz-1.ocr.txt"), "steinitz-2.jpg": fixture("steinitz-2.ocr.txt")}
    [game] = convert(images, FakeReader(texts))
    from tests.conftest import read
    assert_exact_or_flagged(list(read(game.pgn).mainline_moves()), expected_moves("steinitz.pgn"), game.flags)


def test_result_from_white_resigns():
    assert build_game(fixture("fox.ocr.txt")).decoded.result == "0-1"


# --- routine typography is not a doubt -------------------------------------------------------------

def test_typographic_captures_not_flagged():
    flags = build_game(fixture("fox.ocr.txt")).flags
    assert not [f for f in flags if "×" in f.token], flags


def test_l_for_1_in_squares_not_flagged():
    flags = build_game(fixture("mason.ocr.txt")).flags
    assert not [f for f in flags if f.token in {"Rael", "Rgl", "Rbl", "Rdl"}], flags


def test_braced_commentary_is_never_a_side_line():
    built = build_game(fixture("mason.ocr.txt"))
    assert not [f for f in built.flags if f.reason.startswith("side:")]


def test_resigns_alone_gives_result():
    text = fixture("steinitz-1.ocr.txt") + "\n" + fixture("steinitz-2.ocr.txt")
    assert build_game(text).decoded.result == "1-0"


def test_few_flags_on_clean_paragraph_books():
    for name in ("fox", "mason"):
        assert len(build_game(fixture(f"{name}.ocr.txt")).flags) <= 2
