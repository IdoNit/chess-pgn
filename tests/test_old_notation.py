"""Old English descriptive typography: Kt for the knight, sq for the first rank, spaced squares, o-o, and
parenthesised squares that say which piece moved or where it captured."""

import chess
import pytest

from chesspgn.decoder import decode_text
from chesspgn.tokens import normalize_notation, tokenize


def texts(line: str) -> list[str]:
    return [t.text for t in tokenize(line) if t.text]


@pytest.mark.parametrize("line,expected", [
    ("1. P — Q 4 P — Q 4", ["P-Q4", "P-Q4"]),
    ("3. Kt — Q B 3 Kt — K B 3", ["N-QB3", "N-KB3"]),
    ("16. K R - Kt 1 Kt - Q 2", ["KR-N1", "N-Q2"]),
    ("10. P × P P × P", ["PxP", "PxP"]),
    ("9 B × B P, P — Q Kt 3", ["BxBP", "P-QN3"]),
    ("27. B - K 6 ch K - B 1", ["B-K6ch", "K-B1"]),
    ("25. P × P e.p. Kt × P (B 3)", ["PxPe.p.", "NxP@B3"]),
    ("25. R (K1) - Q B 1 K - Kt 2", ["R@K1-QB1", "K-N2"]),
    ("12. O - O - O O - O - O", ["O-O-O", "O-O-O"]),
    ("8. o—o P—Q4", ["O-O", "P-Q4"]),
    ("53. R—Ktsq K—Rsq", ["R-N1", "K-R1"]),
    ("21. B—K3 R(Bsq)—Bsq", ["B-K3", "R@B1-B1"]),
    ("15. Kt×Ktch B×Kt", ["NxNch", "BxN"]),
    ("9. Kt(Q4)—\nK2 R—Ksq", ["N@Q4-K2", "R-K1"]),
    ("20. R × Q R - Q 7", ["RxQ", "R-Q7"]),
    ("14. B × B P × B", ["BxB", "PxB"]),  # not "BxBP" and a stray "x B"
    ("15. K t × K t P - K t 4", ["NxN", "P-N4"]),
    ("20. Q R—Q I P—K Kt 4", ["QR-Q1", "P-KN4"]),  # rank 1 read as I  # the reader sometimes spaces out "Kt"
    # ... and sometimes writes a line as LaTeX math
    (r"$13. B \times B \quad K \times B$", ["BxB", "KxB"]),
    (r"8. $Kt - B_3$ $P - Q Kt_3$", ["N-B3", "P-QN3"]),  # not "RxQR" followed by a stray "- Q 7"
    ("21. Q × P Q R - Kt 1", ["QxP", "QR-N1"]),
])
def test_old_typography_becomes_one_token_per_move(line, expected):
    assert texts(line)[-len(expected):] == expected


def test_prose_is_left_alone():
    line = "The White Knight is practically pinned, because he does not dare move."
    assert normalize_notation(line) == line


def test_normalizing_twice_changes_nothing():
    once = normalize_notation("3. Kt — Q B 3 Kt — K B 3")
    assert normalize_notation(once) == once


def san(decoded) -> list[str]:
    assert decoded.flags == [], "read exactly as printed: no correction should be needed"
    board, out = chess.Board(), []
    for move in decoded.moves:
        out.append(board.san(move))
        board.push(move)
    return out


def test_a_spaced_old_opening_decodes():
    text = "1. P — Q 4 P — Q 4\n2. P — Q B 4 P — K 3\n3. Kt — Q B 3 Kt — K B 3\n4. B — Kt 5 B — K 2"
    assert san(decode_text(text)) == ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7"]


def test_a_pawn_named_by_its_file_captures():
    assert san(decode_text("1. P-K4 P-Q4 2. KPxP QxP"))[2:] == ["exd5", "Qxd5"]


def test_a_parenthesised_square_says_which_knight_moved():
    text = "1. P-K4 P-K4 2. Kt-KB3 Kt-QB3 3. P-Q4 PxP 4. KtxP Kt-B3 5. Kt-QB3 B-K2 6. Kt(Q4)-K2"
    assert san(decode_text(text))[-1] == "Nde2"


def test_a_parenthesised_capture_square_may_be_named_from_either_side():
    # Lasker 1909 writes Black's capture on f3 as "BxKt (B3)": the knight's square counted from White's side.
    text = ("1. P-K4 P-K4 2. Kt-KB3 Kt-QB3 3. Kt-B3 B-Kt5 4. P-Q3 P-Q3 5. B-Q2 B-Kt5 "
            "6. B-K2 BxKt(KB3)")
    assert san(decode_text(text))[-1] == "Bxf3"


# --- analysis written as a paragraph of moves --------------------------------------------------------

def test_moves_separated_by_semicolons_are_analysis_not_the_game():
    from chesspgn.layout import mark_commentary

    analysis = "21 N-K7ch K-B1; 22 R-B7 R-K1; 23 RxB, RxN; 24 R-N8ch, R-K1; 25 RxRch, KxR, and White will draw."
    assert mark_commentary(f"21. R-N1\n\n{analysis}\n\n21. ... R-K1\n") == f"21. R-N1\n\n{{{analysis}}}\n\n21. ... R-K1\n"


def test_a_semicolon_inside_a_bracketed_side_line_changes_nothing():
    from chesspgn.layout import mark_commentary

    line = "5 Nc3 [5...a5 6 Bd3 h6\n7 Bxh6 cb 8 Bxg7!; 5...Na5!] 5...Nf6 6"
    assert mark_commentary(line) == line


# --- a whole move missing from the text ---------------------------------------------------------------

def test_a_move_number_that_jumps_by_one_inserts_a_flagged_pair_of_moves():
    # The reader dropped the line "2. Nf3 Nc6" (it happens next to running heads and diagrams).
    decoded = decode_text("1. e4 e5\n3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O")
    board, out = chess.Board(), []
    for move in decoded.moves:
        out.append(board.san(move))
        board.push(move)
    assert out[4:] == ["Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5", "Bb3", "d6", "c3", "O-O"]
    assert {f.ply for f in decoded.flags if f.reason == "missing"} == {3, 4}


# --- a printed move that fits two moves, decided wrongly, shows up as an error later --------------------

def test_an_ambiguous_move_is_flagged_when_trouble_follows():
    # "11. B-R4" fits Ba4 and Bh4; the move that would tell them apart (14... KtxB) is unreadable here.
    text = ("1. P-Q4 P-Q4 2. Kt-KB3 Kt-KB3 3. P-B4 P-K3 4. B-Kt5 B-K2 5. Kt-B3 QKt-Q2 6. P-K3 O-O "
            "7. R-B1 P-QKt3 8. PxP PxP 9. B-Kt5 B-Kt2 10. O-O P-QR3 11. B-R4 R-B1 12. Q-K2 P-B4 "
            "13. PxP KtxP 14. KR-Q1 K%#B 15. KtxKt P-QKt4")
    decoded = decode_text(text)
    assert 21 in {f.ply for f in decoded.flags}


def test_an_ambiguous_move_with_no_trouble_after_it_is_not_flagged():
    text = ("1. P-Q4 P-Q4 2. Kt-KB3 Kt-KB3 3. P-B4 P-K3 4. B-Kt5 B-K2 5. Kt-B3 QKt-Q2 6. P-K3 O-O "
            "7. R-B1 P-QKt3 8. PxP PxP 9. B-Kt5 B-Kt2 10. O-O P-QR3 11. B-R4 R-B1 12. Q-K2 P-B4 "
            "13. PxP KtxP 14. KR-Q1 KtxB 15. KtxKt P-QKt4")
    assert decode_text(text).flags == []


def test_a_black_move_number_with_spaced_dots_stays_a_number():
    from chesspgn.tokens import Token

    assert tokenize("21. ... R-K1")[0] == Token("", number=21, black_to_move=True)
