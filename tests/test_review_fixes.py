"""Regression tests for the final review's Critical and Important findings."""

import time

import chess
from fastapi.testclient import TestClient

from chesspgn.decoder import decode_text
from chesspgn.games import pgn_filename, split_games
from chesspgn.variations import build_game
from tests.conftest import expected_moves, fixture, read
from tests.helpers import render


def razov_text() -> str:
    return render(expected_moves("razov.pgn"), "english")


def plies_flagged(decoded) -> set[int]:
    return {f.ply for f in decoded.flags}


# C1: a garbled piece move with no digit or x must not vanish silently
def test_c1_garbled_piece_move_is_repaired_and_flagged():
    text = razov_text().replace(" Nc6 ", " Nc& ", 1)  # 2...Nc6 is ply 4
    d = decode_text(text, convention="english")
    assert d.moves == expected_moves("razov.pgn")
    assert plies_flagged(d) & {3, 4, 5}


def test_c1_garbled_last_move_is_flagged():
    text = razov_text().rstrip() .rsplit(" ", 1)[0] + " Bc&"  # last move 27.Bc5 garbled
    d = decode_text(text, convention="english")
    assert len(d.moves) == 53 or any(f.ply >= 52 for f in d.flags)
    assert any(f.ply >= 52 for f in d.flags)


# C2: decoding stops at the result; the next game is not appended
def test_c2_main_line_stops_at_result():
    game1 = fixture("razov.letters.txt")
    game2 = fixture("zikov.letters.txt").replace("1 e4 Nc6", "l e4 Nc6")
    d = decode_text(game1 + "\n" + game2, convention="english")
    assert d.moves == expected_moves("razov.pgn")


# I1: an OCR'd "1" (l / I) still starts the game
def test_i1_game_start_with_l_or_i():
    for one in ("l", "I"):
        text = fixture("stein.ocr.txt").replace("1 P–K4", f"{one} P–K4")
        assert len(split_games(text)) == 1
        assert decode_text(text).moves == expected_moves("stein.pgn")


def test_i1_two_games_split_even_with_ocr_one():
    text = fixture("razov.letters.txt") + "\n" + fixture("zikov.letters.txt").replace("1 e4 Nc6", "l e4 Nc6")
    assert len(split_games(text)) == 2


# I2: a move-number mismatch is flagged
def test_i2_number_mismatch_flagged():
    text = fixture("stein.ocr.txt").replace("26 QxP", "36 QxP")
    d = decode_text(text)
    assert d.moves == expected_moves("stein.pgn")
    assert any(f.reason == "number" and f.ply == 51 for f in d.flags)


# I3: side-line flags carry the absolute ply
def test_i3_side_flag_absolute_ply():
    text = fixture("razov.letters.txt").replace("[14 Qxb7? Rb8", "[14 QXb7? Rb8")
    built = build_game(text)
    side = [f for f in built.flags if f.reason.startswith("side:")]
    assert any(f.ply == 27 for f in side)  # 14. Qxb7 is ply 27


# I5: one page-level deadline covers side lines too
def test_i5_page_deadline_covers_side_lines():
    started = time.perf_counter()
    built = build_game(fixture("razov.letters.txt"), time_budget_s=0.01)
    assert time.perf_counter() - started < 20
    assert any(f.reason == "time_budget" for f in built.flags)


# M2 (re-graded): no names ending in "." or space
def test_m2_filename_has_no_trailing_dot():
    assert pgn_filename({"White": "Smith", "Black": "Jones J.", "Site": "?", "Date": "????.??.??"}) == "Smith-Jones J.pgn"


BREYER = "1 e4 e5 2 Nf3 Nc6 3 Bb5 a6 4 Ba4 Nf6 5 0-0 Be7 6 Re1 b5 7 Bb3 d6 8 c3 0-0 9 h3 Nb8 10 d4 Nbd7 11 c4 c6 12 cxb5 axb5 1-0"


def breyer_moves() -> list[chess.Move]:
    board, moves = chess.Board(), []
    for san in "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7 c4 c6 cxb5 axb5".split():
        moves.append(board.push_san(san))
    return moves


def test_c1_reviewer_case_nb_ampersand():
    d = decode_text(BREYER.replace("9 h3 Nb8", "9 h3 Nb&"), convention="english")
    wrong = next((i for i, (a, b) in enumerate(zip(d.moves, breyer_moves()), start=1) if a != b), None)
    if d.moves != breyer_moves():
        assert any(abs(f.ply - (wrong or len(d.moves))) <= 1 for f in d.flags), (d.moves, d.flags)
    assert d.moves == breyer_moves()
