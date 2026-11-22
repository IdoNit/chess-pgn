import time

import chess

from chesspgn import decoder
from chesspgn.decoder import decode_text
from tests.conftest import expected_moves, fixture


def test_convention_committed_early(monkeypatch):
    full_runs: list[str] = []
    original = decoder._decode_with_convention

    def spy(tokens, convention, start, stop=None, **kwargs):
        if stop is None:
            full_runs.append(convention)
        return original(tokens, convention, start, stop, **kwargs)

    monkeypatch.setattr(decoder, "_decode_with_convention", spy)
    d = decode_text(fixture("stein.ocr.txt"))
    assert d.moves == expected_moves("stein.pgn")
    assert full_runs == ["descriptive"]


def test_stein_under_15s():
    started = time.perf_counter()
    decode_text(fixture("stein.ocr.txt"))
    assert time.perf_counter() - started < 15


def test_time_budget_returns_flagged():
    d = decode_text(fixture("stein.ocr.txt"), time_budget_s=0.01)
    board = chess.Board()
    for move in d.moves:
        assert board.is_legal(move)
        board.push(move)
    assert any(f.reason == "time_budget" for f in d.flags)
