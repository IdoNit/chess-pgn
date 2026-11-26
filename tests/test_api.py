from pathlib import Path
from urllib.parse import unquote

import chess
import pytest
from fastapi.testclient import TestClient

from chesspgn.reader import ReaderUnavailable
from chesspgn.web.app import create_app
from tests.conftest import expected_moves, fixture, fixture_path, read


class SequenceReader:
    def __init__(self, *texts: str):
        self.texts = list(texts)

    def read(self, image: Path) -> str:
        return self.texts.pop(0)


class DownReader:
    def read(self, image: Path) -> str:
        raise ReaderUnavailable("down")


def upload(client: TestClient, *names: str):
    files = [("images", (name, fixture_path(name).read_bytes(), "image/jpeg")) for name in names]
    return client.post("/api/convert", files=files)


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    client = TestClient(create_app(tmp_path_factory.mktemp("data"), SequenceReader(fixture("stein.ocr.txt"))))
    response = upload(client, "stein.jpg")
    return client, response


def test_convert_returns_game(converted):
    _client, response = converted
    assert response.status_code == 200
    [game] = response.json()["games"]
    assert game["filename"] == "Stein-Korchnoi - Moscow 1964.pgn" and game["move_count"] == 93


def test_list_and_get(converted):
    client, response = converted
    game_id = response.json()["games"][0]["id"]
    assert [g["id"] for g in client.get("/api/games").json()] == [game_id]
    detail = client.get(f"/api/games/{game_id}").json()
    assert list(read(detail["pgn"]).mainline_moves()) == expected_moves("stein.pgn")
    assert detail["moves"][48]["san"] == "Bxf6" and detail["moves"][48]["ply"] == 49
    assert client.get(detail["images"][0]).status_code == 200


def test_pin_redecodes(converted):
    client, response = converted
    game_id = response.json()["games"][0]["id"]
    detail = client.post(f"/api/games/{game_id}/pin", json={"ply": 49, "uci": "d3g6"}).json()
    game = read(detail["pgn"])
    assert game.errors == []
    assert list(game.mainline_moves())[48] == chess.Move.from_uci("d3g6")
    assert detail["pins"] == {"49": "d3g6"}


def test_headers_and_download(converted):
    client, response = converted
    game_id = response.json()["games"][0]["id"]
    detail = client.put(f"/api/games/{game_id}/headers", json={"White": "Stein, Leonid"}).json()
    assert detail["headers"]["White"] == "Stein, Leonid"
    download = client.get(f"/api/games/{game_id}/download")
    assert download.status_code == 200 and '[White "Stein, Leonid"]' in download.text
    assert "Stein, Leonid-Korchnoi - Moscow 1964.pgn" in unquote(download.headers["content-disposition"])


def test_no_game_is_422(tmp_path):
    client = TestClient(create_app(tmp_path, SequenceReader("Chapter 3. Some prose only.")))
    response = upload(client, "stein.jpg")
    assert response.status_code == 422 and response.json() == {"error": "no_game"}


def test_engine_down_is_503(tmp_path):  # Review Focus 3
    client = TestClient(create_app(tmp_path, DownReader()))
    response = upload(client, "stein.jpg")
    assert response.status_code == 503 and response.json() == {"error": "engine_down"}


def test_suggestions_for_flags(tmp_path):
    text = fixture("stein.ocr.txt").replace("47 QxP+", "47 QXP+")
    client = TestClient(create_app(tmp_path, SequenceReader(text)))
    game_id = upload(client, "stein.jpg").json()["games"][0]["id"]
    detail = client.get(f"/api/games/{game_id}").json()
    [flag] = [f for f in detail["flags"] if f["ply"] == 93]
    assert flag["chosen"] == "Qxf7+" and 1 <= len(flag["suggestions"]) <= 3 and "Qxf7+" not in flag["suggestions"]


def test_index_served(tmp_path):
    client = TestClient(create_app(tmp_path, DownReader()))
    response = client.get("/")
    assert response.status_code == 200 and 'dir="rtl"' in response.text


def test_delete_game(tmp_path):
    client = TestClient(create_app(tmp_path, SequenceReader(fixture("stein.ocr.txt"))))
    game_id = upload(client, "stein.jpg").json()["games"][0]["id"]
    assert client.delete(f"/api/games/{game_id}").status_code == 200
    assert client.get("/api/games").json() == []
    assert client.get(f"/api/games/{game_id}").status_code == 404


# C3: illegal pins are refused; pins that no longer apply are dropped and reported
def test_c3_illegal_pin_refused(tmp_path):
    client = TestClient(create_app(tmp_path, SequenceReader(fixture("stein.ocr.txt"))))
    game_id = upload(client, "stein.jpg").json()["games"][0]["id"]
    for body in ({"ply": 49, "uci": "a1a8"}, {"ply": 999, "uci": "e2e4"}, {"ply": "x", "uci": "zz"}, {}):
        response = client.post(f"/api/games/{game_id}/pin", json=body)
        assert response.status_code == 422 and response.json() == {"error": "illegal_pin"}


def test_c3_unhonoured_pin_dropped_and_flagged(tmp_path):
    from chesspgn.storage import GameStore

    client = TestClient(create_app(tmp_path, SequenceReader(fixture("stein.ocr.txt"))))
    game_id = upload(client, "stein.jpg").json()["games"][0]["id"]
    store = GameStore(tmp_path)
    _record, state = store.load(game_id)
    state["pins"] = {"60": "a1a8"}  # stale pin that cannot be played
    store.update(game_id, state)
    detail = client.post(f"/api/games/{game_id}/pin", json={"ply": 49, "uci": "d3g6"}).json()
    assert detail["pins"] == {"49": "d3g6"}
    assert any(f["ply"] == 60 and f["reason"] == "pin_dropped" for f in detail["flags"])
    assert detail["move_count"] > 60


# I3: a side-line flag can be confirmed (dismissed)
def test_i3_dismiss_flag(tmp_path):
    text = fixture("stein.ocr.txt").replace("47 QxP+", "47 QXP+")
    client = TestClient(create_app(tmp_path, SequenceReader(text)))
    game_id = upload(client, "stein.jpg").json()["games"][0]["id"]
    detail = client.post(f"/api/games/{game_id}/dismiss", json={"ply": 93, "reason": "ocr"}).json()
    assert not any(f["ply"] == 93 for f in detail["flags"])
    assert client.get("/api/games").json()[0]["flags_open"] == 0


# I4: several games download as one combined PGN
def test_i4_combined_download(tmp_path):
    client = TestClient(create_app(tmp_path, SequenceReader(fixture("razov.letters.txt") + "\n" + fixture("zikov.letters.txt"))))
    games = upload(client, "razov.jpg").json()["games"]
    assert len(games) == 2
    ids = ",".join(g["id"] for g in games)
    response = client.get(f"/api/download?ids={ids}")
    assert response.status_code == 200 and response.text.count("[Event ") == 2


class CutOffReader:
    def read(self, image: Path) -> str:
        from chesspgn.reader import PageIncomplete

        raise PageIncomplete("cut off")


def test_cut_off_page_is_422(tmp_path):
    client = TestClient(create_app(tmp_path, CutOffReader()))
    response = upload(client, "stein.jpg")
    assert response.status_code == 422 and response.json() == {"error": "page_incomplete"}


# A confirmed flag stays confirmed after the user fixes another move (the game is decoded again then).
def test_confirmation_survives_a_later_fix(tmp_path):
    text = fixture("stein.ocr.txt").replace("47 QxP+", "47 QXP+")
    client = TestClient(create_app(tmp_path, SequenceReader(text)))
    game_id = upload(client, "stein.jpg").json()["games"][0]["id"]
    client.post(f"/api/games/{game_id}/dismiss", json={"ply": 93, "reason": "ocr"})
    book_move = expected_moves("stein.pgn")[48].uci()
    detail = client.post(f"/api/games/{game_id}/pin", json={"ply": 49, "uci": book_move}).json()
    assert detail["moves"][92]["san"] == "Qxf7+" and not any(f["ply"] == 93 for f in detail["flags"])


# When the page text ends before the game does, the user adds the remaining moves one by one.
def test_moves_added_after_the_end_are_kept(tmp_path):
    text = fixture("stein.ocr.txt").split("\n43...KxB")[0]
    book = expected_moves("stein.pgn")
    client = TestClient(create_app(tmp_path, SequenceReader(text)))
    game_id = upload(client, "stein.jpg").json()["games"][0]["id"]
    detail = client.get(f"/api/games/{game_id}").json()
    end = len(detail["moves"])
    for ply in (end + 1, end + 2):
        detail = client.post(f"/api/games/{game_id}/pin", json={"ply": ply, "uci": book[ply - 1].uci()}).json()
    assert [m["uci"] for m in detail["moves"]] == [m.uci() for m in book[: end + 2]]
    assert not any(f["reason"] == "pin_dropped" for f in detail["flags"])


# The decoder can run past the real end of a game into the text after it: the user ends the game at a move.
def test_end_the_game_at_a_move(tmp_path):
    client = TestClient(create_app(tmp_path, SequenceReader(fixture("stein.ocr.txt"))))
    game_id = upload(client, "stein.jpg").json()["games"][0]["id"]
    detail = client.post(f"/api/games/{game_id}/end", json={"ply": 60}).json()
    assert len(detail["moves"]) == 60 and detail["end"] == 60
    assert all(f["ply"] <= 60 for f in detail["flags"])
    assert "61." not in client.get(f"/api/games/{game_id}/download").text.split("\n\n", 1)[1].split(" (")[0]


def test_the_end_stays_after_a_later_fix_and_can_be_undone(tmp_path):
    client = TestClient(create_app(tmp_path, SequenceReader(fixture("stein.ocr.txt"))))
    game_id = upload(client, "stein.jpg").json()["games"][0]["id"]
    full = len(client.get(f"/api/games/{game_id}").json()["moves"])
    client.post(f"/api/games/{game_id}/end", json={"ply": 60})
    book_move = expected_moves("stein.pgn")[48].uci()
    assert len(client.post(f"/api/games/{game_id}/pin", json={"ply": 49, "uci": book_move}).json()["moves"]) == 60
    detail = client.post(f"/api/games/{game_id}/end", json={"ply": None}).json()
    assert len(detail["moves"]) == full and detail["end"] is None


def test_a_move_added_after_the_end_moves_the_end(tmp_path):
    client = TestClient(create_app(tmp_path, SequenceReader(fixture("stein.ocr.txt"))))
    game_id = upload(client, "stein.jpg").json()["games"][0]["id"]
    client.post(f"/api/games/{game_id}/end", json={"ply": 60})
    detail = client.post(f"/api/games/{game_id}/pin", json={"ply": 61, "uci": expected_moves("stein.pgn")[60].uci()}).json()
    assert len(detail["moves"]) == 61 and detail["end"] == 61


# Where each move is printed (page and row box), for the review screen's page viewer.
def test_places_give_each_move_its_page_and_row(tmp_path, monkeypatch):
    from chesspgn import repair

    calls = []

    def lines(image):
        calls.append(1)
        return [("1. P-K4 P-K4", (10, 20, 300, 40))]

    monkeypatch.setattr(repair, "page_lines", lines)
    client = TestClient(create_app(tmp_path, SequenceReader(fixture("stein.ocr.txt"))))
    game_id = upload(client, "stein.jpg").json()["games"][0]["id"]
    moves = len(client.get(f"/api/games/{game_id}").json()["moves"])
    places = client.get(f"/api/games/{game_id}/places").json()["places"]
    assert len(places) == moves and places[0]["page"] == 0 and places[0]["box"] is not None
    assert places[2]["box"] is None  # move 2 is not among tesseract's lines here
    client.get(f"/api/games/{game_id}/places")
    assert len(calls) == 1  # read once, then kept


# After an update the browser must not keep running the old script: the app's files are always revalidated.
def test_the_apps_own_files_are_not_served_stale(tmp_path):
    client = TestClient(create_app(tmp_path, SequenceReader()))
    for path in ("/", "/static/app.js", "/static/strings.he.json"):
        response = client.get(path)
        assert response.status_code == 200 and response.headers.get("cache-control") == "no-cache", path
