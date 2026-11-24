"""Measuring the reader on its own: is each book move in the text, at its numbered place?"""

from bench.extraction import extraction


def test_every_move_printed_cleanly_is_exact():
    result = extraction("1. P-K4 P-K4\n2. Kt-KB3 Kt-QB3", ["e4", "e5", "Nf3", "Nc6"])
    assert (result.exact, result.fixable, result.damaged, result.missing) == (4, 0, 0, 0)


def test_a_move_one_ocr_correction_away_is_fixable():
    result = extraction("1. P-K8 P-K4", ["e3", "e5"])
    assert (result.exact, result.fixable) == (1, 1)


def test_a_garbled_move_is_damaged():
    result = extraction("1. P-K4 P-K4\n2. Kt-KB3 K%#Q@3", ["e4", "e5", "Nf3", "Nc6"])
    assert result.damaged == 1 and result.bad_plies == [4]


def test_a_dropped_line_is_missing():
    result = extraction("1. P-K4 P-K4\n3. B-Kt5 P-QR3", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"])
    assert (result.exact, result.missing) == (4, 2)


def test_a_black_move_on_its_own_row_counts():
    result = extraction("1. P-K4\nA note about the move.\n1. ... P-K4", ["e4", "e5"])
    assert result.exact == 2


def test_old_typography_is_read():
    result = extraction("1. P — Q 4 Kt — K B 3", ["d4", "Nf6"])
    assert result.exact == 2
