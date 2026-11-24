"""Rows the board couldn't read get a second reading from a vision model; those moves are flagged "read by AI"."""

import chess

from chesspgn.decoder import decode_text
from chesspgn.tokens import tokenize


def san(decoded) -> list[str]:
    board, out = chess.Board(), []
    for move in decoded.moves:
        out.append(board.san(move))
        board.push(move)
    return out


# --- task 1: the marker ---------------------------------------------------------------------------------

def test_a_move_marked_with_a_dagger_is_an_ai_reading():
    token = [t for t in tokenize("2. N-KB3† N-QB3") if t.text][0]
    assert (token.text, token.ai) == ("N-KB3", True)


def test_an_ai_reading_is_flagged_even_when_it_reads_cleanly():
    decoded = decode_text("1. P-K4 P-K4 2. Kt-KB3† Kt-QB3")
    assert san(decoded) == ["e4", "e5", "Nf3", "Nc6"]
    assert [(f.ply, f.reason) for f in decoded.flags] == [(3, "ai_read")]


# --- task 2: rewriting a damaged row with the model's reading ----------------------------------------------

from chesspgn.repair import repaired_line  # noqa: E402


def test_a_row_that_reads_cleanly_is_not_rewritten():
    assert repaired_line("19. P × P B × P", 19, ("P × P", "B × P")) is None


def test_a_damaged_row_takes_a_close_reading_marked_as_ai():
    # the reader glued the wrapped "B" of White's move onto Black's: "Kt (B3)× / B   P—B4"
    line = repaired_line("18. Kt (B3)×P—B4", 18, ("Kt (B3)×B", "P—B4"))
    assert line == "18. Kt (B3)×B† P—B4†"


def test_a_reading_that_is_nothing_like_the_ocr_is_refused():
    assert repaired_line("18. Kt (B3)×P—B4", 18, ("Q—R 5", "K—Kt 1")) is None


def test_a_reading_that_is_not_moves_is_refused():
    assert repaired_line("18. Kt (B3)×P—B4", 18, ("Kt (B3)×", "and so on")) is None


def test_a_black_row_stays_a_black_row():
    assert repaired_line("21. ... R % K 1", 21, ("", "R — K 1")) == "21. ... R — K 1†"


# --- task 3: the second pass in the pipeline --------------------------------------------------------------

from PIL import Image  # noqa: E402

from chesspgn import repair  # noqa: E402
from chesspgn.pipeline import convert  # noqa: E402

PAGE = "1. P-K4 P-K4\n2. Kt-KB3 K%#QB3\n3. B-Kt5 P-QR3\n4. B-R4 Kt-B3\n"


class OnePage:
    def read(self, image):
        return PAGE


class RowReaderStub:
    def __init__(self, answers):
        self.answers, self.asked = answers, []

    def read_row(self, strip, number):
        self.asked.append(number)
        return self.answers.get(number)


def blank_page(tmp_path, monkeypatch):
    path = tmp_path / "page.png"
    Image.new("RGB", (400, 300), "white").save(path)
    monkeypatch.setattr(repair, "page_lines", lambda image: [(line, (10, 20 * k, 300, 20 * k + 15))
                                                           for k, line in enumerate(PAGE.splitlines())])
    return path


def test_without_the_helper_the_damaged_move_stays_flagged(tmp_path, monkeypatch):
    [game] = convert([blank_page(tmp_path, monkeypatch)], OnePage())
    assert any(f.ply == 4 and f.reason != "ai_read" for f in game.flags)


def test_the_helper_is_asked_only_about_the_damaged_row_and_its_moves_are_flagged_as_ai(tmp_path, monkeypatch):
    rows = RowReaderStub({2: ("Kt — K B 3", "Kt — Q B 3")})
    [game] = convert([blank_page(tmp_path, monkeypatch)], OnePage(), row_reader=rows)
    assert rows.asked == [2]
    assert "4. Ba4 Nf6" in game.pgn and "2. Nf3 Nc6" in game.pgn
    assert {(f.ply, f.reason) for f in game.flags} == {(3, "ai_read"), (4, "ai_read")}


def test_a_reading_the_board_rejects_leaves_the_flag(tmp_path, monkeypatch):
    rows = RowReaderStub({2: ("Kt — K B 3", "Q — R 5")})  # Qh4: legal, but nothing like the OCR's row
    [game] = convert([blank_page(tmp_path, monkeypatch)], OnePage(), row_reader=rows)
    assert not any(f.reason == "ai_read" for f in game.flags)


def test_progress_is_reported_while_the_helper_works(tmp_path, monkeypatch):
    seen = []
    convert([blank_page(tmp_path, monkeypatch)], OnePage(), row_reader=RowReaderStub({}), progress=seen.append)
    assert seen and any("1/1" in s for s in seen)


# --- task 4: the setting, the progress line and the strings in the app ----------------------------------------

from fastapi.testclient import TestClient  # noqa: E402

from chesspgn.web.app import create_app  # noqa: E402
from tests.conftest import fixture, fixture_path  # noqa: E402


class StubPages:
    def __init__(self, text):
        self.text = text

    def read(self, image):
        return self.text


def app_client(tmp_path, memory_gb, row_reader=None):
    return TestClient(create_app(tmp_path, StubPages(fixture("stein.ocr.txt")), row_reader=row_reader,
                                 memory_gb=memory_gb))


def test_the_ai_helper_is_on_by_default_with_enough_memory(tmp_path):
    settings = app_client(tmp_path, 32, RowReaderStub({})).get("/api/settings").json()
    assert settings == {"ai_helper": True, "available": True, "enough_memory": True, "memory_gb": 32}


def test_the_ai_helper_is_off_by_default_without_enough_memory_but_can_be_switched_on(tmp_path):
    client = app_client(tmp_path, 8, RowReaderStub({}))
    assert client.get("/api/settings").json()["ai_helper"] is False
    assert client.put("/api/settings", json={"ai_helper": True}).json()["ai_helper"] is True
    assert app_client(tmp_path, 8, RowReaderStub({})).get("/api/settings").json()["ai_helper"] is True  # kept


def test_the_ai_helper_is_unavailable_without_a_row_reader(tmp_path):
    assert app_client(tmp_path, 32).get("/api/settings").json() | {} == {
        "ai_helper": False, "available": False, "enough_memory": True, "memory_gb": 32}


def test_switched_off_the_helper_is_never_asked(tmp_path):
    rows = RowReaderStub({})
    client = app_client(tmp_path, 32, rows)
    client.put("/api/settings", json={"ai_helper": False})
    files = [("images", ("stein.jpg", fixture_path("stein.jpg").read_bytes(), "image/jpeg"))]
    assert client.post("/api/convert", files=files).status_code == 200
    assert rows.asked == []


def test_progress_is_empty_when_idle(tmp_path):
    assert app_client(tmp_path, 32).get("/api/progress").json() == {"step": ""}
