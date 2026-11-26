"""Game headings (players, event, year) read by the AI helper, kept only where the heading prints them."""

import httpx
from PIL import Image

from chesspgn import repair
from chesspgn.games import extract_headers
from chesspgn.pipeline import convert

PAGE = """GAME 1. QUEEN'S GAMBIT DECLINED
(Match, 1909)

White: F. J. Marshall. Black: J. R. Capablanca.

1. P-Q4 P-Q4
2. P-QB4 P-K3
3. Kt-QB3 Kt-KB3
"""


class OnePage:
    def read(self, image):
        return PAGE


class Helper:
    def __init__(self, answer):
        self.answer, self.asked = answer, []

    def read_row(self, strip, number):
        return None

    def read_headers(self, heading):
        self.asked.append(heading)
        return self.answer


def page(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (400, 300), "white").save(path)
    return path


def test_the_helper_reads_the_players_from_the_heading(tmp_path):
    helper = Helper({"white": "F. J. Marshall", "black": "J. R. Capablanca", "event": "Match", "site": "",
                     "year": "1909"})
    [game] = convert([page(tmp_path)], OnePage(), row_reader=helper)
    assert game.headers["White"] == "F. J. Marshall" and game.headers["Black"] == "J. R. Capablanca"
    assert game.headers["Event"] == "Match" and game.headers["Date"].startswith("1909")
    assert '[White "F. J. Marshall"]' in game.pgn and "Marshall" in game.filename
    assert "White: F. J. Marshall." in helper.asked[0]


def test_a_name_the_heading_does_not_print_is_not_used(tmp_path):
    helper = Helper({"white": "Emanuel Lasker", "black": "J. R. Capablanca", "event": "", "site": "", "year": "1921"})
    [game] = convert([page(tmp_path)], OnePage(), row_reader=helper)
    assert game.headers["White"] != "Emanuel Lasker" and game.headers["Black"] == "J. R. Capablanca"
    assert not game.headers["Date"].startswith("1921")


def test_without_the_helper_the_event_has_no_stray_bracket():
    assert "(" not in extract_headers(PAGE)["Event"] and ")" not in extract_headers(PAGE)["Event"]


def test_the_ollama_helper_answers_the_heading_question_in_json():
    def answer(content):
        return httpx.MockTransport(lambda request: httpx.Response(200, json={"message": {"content": content}}))

    reader = repair.OllamaRowReader("http://x", transport=answer('{"white": "Marshall", "black": "Capablanca"}'))
    assert reader.read_headers("White: Marshall. Black: Capablanca.")["white"] == "Marshall"
    assert repair.OllamaRowReader("http://x", transport=answer("nope")).read_headers("x") is None


def test_a_name_or_place_made_of_digits_is_not_used(tmp_path):
    helper = Helper({"white": "F. J. Marshall", "black": "J. R. Capablanca", "event": "Match", "site": "1909",
                     "year": "1909"})
    [game] = convert([page(tmp_path)], OnePage(), row_reader=helper)
    assert game.headers["Site"] != "1909"


def test_the_opening_the_heading_names_goes_to_the_opening_tag(tmp_path):
    helper = Helper({"white": "F. J. Marshall", "black": "J. R. Capablanca", "event": "Match", "site": "",
                     "year": "1909", "opening": "QUEEN'S GAMBIT DECLINED"})
    [game] = convert([page(tmp_path)], OnePage(), row_reader=helper)
    assert game.headers["Opening"] == "QUEEN'S GAMBIT DECLINED" and game.headers["Event"] == "Match"
