import pytest

from chesspgn.decoder import Flag
from chesspgn.pipeline import ConvertedGame
from chesspgn.storage import GameNotFound, GameStore
from tests.conftest import fixture, fixture_path


def stein_game(flags: list[Flag] | None = None) -> ConvertedGame:
    return ConvertedGame(
        pgn=fixture("stein.pgn"),
        headers={"White": "Stein", "Black": "Korchnoi", "Site": "Moscow", "Date": "1964.??.??"},
        flags=flags or [],
        move_count=93,
        continues=False,
        filename="Stein-Korchnoi - Moscow 1964.pgn",
        text="1 P-K4 P-QB4",
    )


def test_save_layout(tmp_path):
    record = GameStore(tmp_path).save(stein_game(), [fixture_path("stein.jpg")])
    assert (record.folder / "game.pgn").read_text(encoding="utf-8") == fixture("stein.pgn")
    assert (record.folder / "stein.jpg").exists()
    assert (record.folder / "text.txt").read_text(encoding="utf-8") == "1 P-K4 P-QB4"
    assert record.folder.name.endswith(" Stein-Korchnoi")
    assert record.filename == "Stein-Korchnoi - Moscow 1964.pgn"


def test_same_game_twice_gets_suffix(tmp_path):  # Review Focus 5
    store = GameStore(tmp_path)
    a = store.save(stein_game(), [fixture_path("stein.jpg")])
    b = store.save(stein_game(), [fixture_path("stein.jpg")])
    assert a.folder != b.folder and b.folder.name.endswith("(2)")
    assert (a.folder / "game.pgn").read_text(encoding="utf-8") == fixture("stein.pgn")


def test_list_newest_first(tmp_path):
    store = GameStore(tmp_path)
    first = store.save(stein_game(), [fixture_path("stein.jpg")])
    second = store.save(stein_game(), [fixture_path("stein.jpg")])
    assert [r.id for r in store.list()] == [second.id, first.id]


def test_load_and_update(tmp_path):
    store = GameStore(tmp_path)
    record = store.save(stein_game([Flag(93, "QXP+", "Qxf7+", "ocr")]), [fixture_path("stein.jpg")])
    loaded, state = store.load(record.id)
    assert loaded.flags_open == 1 and state["flags"][0]["chosen"] == "Qxf7+" and state["pins"] == {}
    state["pgn"] = "new pgn\n"
    state["flags"] = []
    updated = store.update(record.id, state)
    assert (record.folder / "game.pgn").read_text(encoding="utf-8") == "new pgn\n"
    assert updated.flags_open == 0


def test_unknown_or_escaping_id(tmp_path):
    store = GameStore(tmp_path / "games")
    with pytest.raises(GameNotFound):
        store.load("../etc")
    with pytest.raises(GameNotFound):
        store.load("nope")
