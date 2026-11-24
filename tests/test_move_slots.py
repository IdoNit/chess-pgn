"""Printed move numbers anchor every move: a move played away from its numbered place is flagged, and a
number that goes back (analysis printed like game text) is not the game."""

import chess
import pytest

from chesspgn.decoder import decode_text


def san(decoded) -> list[str]:
    board, out = chess.Board(), []
    for move in decoded.moves:
        out.append(board.san(move))
        board.push(move)
    return out


MAIN = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5"]


@pytest.mark.xfail(reason="analysis printed like game rows can still pull the decoder off the game: next step", strict=True)
def test_numbered_analysis_between_game_lines_is_not_followed():
    text = ("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6\n3 Bc4 Bc5 4 c3 Nf6 5 d4 exd4 6 cxd4 Bb4+\n"
            "4. Ba4 Nf6 5. O-O Be7 6. Re1 b5")
    assert san(decode_text(text)) == MAIN


def test_a_move_away_from_its_number_is_never_silent():
    # Whatever the decoder makes of a stray line, every move it plays out of its numbered place is flagged.
    text = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6\n3 Bc4 Bc5 4 c3 Nf6\n4. Ba4 Nf6 5. O-O Be7 6. Re1 b5"
    decoded = decode_text(text)
    moves = san(decoded)
    flagged = {f.ply for f in decoded.flags}
    wrong = [i for i, (a, b) in enumerate(zip(moves, MAIN), start=1) if a != b]
    assert all(any(abs(p - w) <= 1 for p in flagged) for w in wrong)


def test_a_black_move_printed_without_dots_is_fine():
    # "14. P-B4" for Black's 14th move: the number is right, only the dots are missing.
    text = "1. e4 e5 2. Nf3 Nc6\n3. Bb5\n3. a6\n4. Ba4 Nf6"
    decoded = decode_text(text)
    assert san(decoded) == MAIN[:8] and decoded.flags == []


def test_a_misprinted_number_costs_flags_not_silent_errors():
    decoded = decode_text("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 3. Ba4 Nf6 5. O-O Be7 6. Re1 b5")
    moves, flagged = san(decoded), {f.ply for f in decoded.flags}
    wrong = [i for i, (a, b) in enumerate(zip(moves, MAIN), start=1) if a != b]
    assert len(moves) == len(MAIN) and all(any(abs(p - w) <= 1 for p in flagged) for w in wrong)


@pytest.mark.xfail(reason="analysis printed like game rows can still pull the decoder off the game: next step", strict=True)
def test_a_note_that_continues_the_numbers_loses_to_the_longer_game():
    # "If 4) ... Bc5; 5) c3 ..." printed as rows right after "4. Ba4": it continues the numbers for a while,
    # but the game itself carries on longer.
    text = ("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4\n4. ... Bc5 5. c3\n"
            "4. ... Nf6 5. O-O Be7 6. Re1 b5")
    assert san(decode_text(text)) == MAIN


# A guessed move (a missing row) decides where later captures land, so the moves that use its squares are
# flagged too, even when their own text reads perfectly.
GUESSED_ROW = "1. e4 e5 2. Nf3 Nc6 4. Bxc6 dxc6 5. O-O f6 6. d4 exd4"


def test_moves_that_use_a_guessed_moves_squares_are_flagged():
    decoded = decode_text(GUESSED_ROW)
    reasons = {f.ply: f.reason for f in decoded.flags}
    assert san(decoded)[6:8] == ["Bxc6", "dxc6"]  # the bishop came from the guessed square, the pawn retakes there
    assert reasons.get(7) == "depends" and reasons.get(8) == "depends"


def test_moves_away_from_a_guess_are_not_flagged():
    decoded = decode_text(GUESSED_ROW)
    assert {f.ply for f in decoded.flags} == {5, 6, 7, 8}


def test_a_fix_on_a_missing_row_is_played():
    # The user fixes a move the decoder had to guess because its row is missing: the fix is always played.
    decoded = decode_text(GUESSED_ROW, pins={6: chess.Move.from_uci("a7a6")})
    assert san(decoded)[4:8] == ["Bb5", "a6", "Bxc6", "dxc6"]


def test_fixes_on_both_moves_of_a_missing_row_are_played():
    pins = {5: chess.Move.from_uci("f1b5"), 6: chess.Move.from_uci("a7a6")}
    assert san(decode_text(GUESSED_ROW, pins=pins))[4:8] == ["Bb5", "a6", "Bxc6", "dxc6"]


def test_a_fix_on_a_single_missing_move_is_played():
    # "3. Bb5 4. Ba4": Black's 3rd move is missing and has to be guessed; the user's fix there is played.
    decoded = decode_text("1. e4 e5 2. Nf3 Nc6 3. Bb5 4. Ba4 Nf6 5. O-O Be7", pins={6: chess.Move.from_uci("d7d6")})
    assert san(decoded)[5:8] == ["d6", "Ba4", "Nf6"]
