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
