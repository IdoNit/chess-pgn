"""Scoring a converted game against the book's expected game."""

from bench.score import score


def test_identical_games_are_exact():
    result = score(["e4", "e5", "Nf3"], ["e4", "e5", "Nf3"], flagged=[])
    assert (result.correct, result.wrong, result.unflagged_wrong, result.exact) == (3, 0, 0, True)


def test_a_wrong_move_with_a_flag_is_flagged_not_silent():
    result = score(["e4", "e5", "Nf3"], ["e4", "e6", "Nf3"], flagged=[2])
    assert (result.correct, result.wrong, result.unflagged_wrong) == (2, 1, 0)


def test_a_wrong_move_without_a_flag_is_silent():
    result = score(["e4", "e5", "Nf3"], ["e4", "e6", "Nf3"], flagged=[])
    assert result.unflagged_wrong == 1
    assert result.silent_plies == [2]


def test_a_flag_one_ply_away_covers_the_error():
    result = score(["e4", "e5", "Nf3", "Nc6"], ["e4", "e6", "Nf3", "Nc6"], flagged=[3])
    assert result.unflagged_wrong == 0


def test_a_missing_move_counts_once_and_later_moves_still_match():
    result = score(["d4", "d5", "c4", "e6", "Nc3"], ["d4", "d5", "c4", "Nc3"], flagged=[4])
    assert (result.correct, result.wrong, result.unflagged_wrong) == (4, 1, 0)


def test_a_missing_move_with_no_flag_nearby_is_silent():
    result = score(["d4", "d5", "c4", "e6", "Nc3"], ["d4", "d5", "c4", "Nc3"], flagged=[])
    assert result.unflagged_wrong == 1


def test_moves_past_the_book_game_are_extra_and_need_a_flag():
    result = score(["e4", "e5"], ["e4", "e5", "Nf3"], flagged=[])
    assert (result.correct, result.extra, result.unflagged_wrong, result.exact) == (2, 1, 1, False)


def test_a_game_that_stops_early_misses_the_rest():
    result = score(["e4", "e5", "Nf3", "Nc6"], ["e4", "e5"], flagged=[])
    assert (result.correct, result.wrong, result.exact) == (2, 2, False)
    assert result.unflagged_wrong == 0  # a short game is visible to the reader; it is not a silent error


def test_accepted_alternatives_count_as_correct():
    result = score(["e4", "e5", "Nf3"], ["e4", "e5", "Nc3"], flagged=[], accept={3: ["Nf3", "Nc3"]})
    assert (result.correct, result.wrong, result.exact) == (3, 0, True)
