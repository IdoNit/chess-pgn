from chesspgn.tokens import Token, tokenize, variants


def test_glued_number():
    assert tokenize("8...Nxd4?") == [Token("", number=8, black_to_move=True), Token("Nxd4", "?")]


def test_white_number_and_annotation():
    assert tokenize("31 RxB!!") == [Token("", number=31), Token("RxB", "!!")]


def test_spaced_dots():
    assert [t.number for t in tokenize("16 . . . N-Q2")] == [16, None]
    assert tokenize("16 . . . N-Q2")[0].black_to_move


def test_dashes_normalised():
    assert tokenize("B–K2")[0].text == "B-K2"
    assert tokenize("R--N3+")[0].text == "R-N3+"


def test_result_kept():
    assert tokenize("27 Bc5! 1:0")[-1].text == "1:0"


def test_trailing_punctuation_stripped():
    assert tokenize("a6; 9 Be2")[0].text == "a6"


def test_variants_start_with_token_itself():
    assert variants("Nf3")[0] == ("Nf3", 0.0)


def test_variants_include_x_insertion():
    assert ("PxP", 1.0) in variants("PP")


def test_variants_drop_group_first():
    assert variants("②c4")[1] == ("c4", 0.8)


def test_variants_confusions_and_insertions():
    found = dict(variants("QXP+"))
    assert found["QxP+"] == 1.0
    assert dict(variants("xN"))["RxN"] == 1.2


def test_variants_bounded():
    assert len(variants("QR-KB1+")) < 2500


def test_black_move_number_before_line_break():
    assert tokenize("8...\nNxd4?") == [Token("", number=8, black_to_move=True), Token("Nxd4", "?")]
