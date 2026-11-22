import chess

from chesspgn.notation import CONVENTIONS, GROUP_SYMBOLS, match_cost, read_clues

START = chess.STARTING_FEN


def legal(fen: str, token: str, convention: str) -> set[str]:
    board = chess.Board(fen)
    clues = read_clues(token, convention)
    if clues is None:
        return set()
    return {board.san(m) for m in board.legal_moves if match_cost(board, m, clues) is not None}


def after(*sans: str) -> str:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board.fen()


def test_conventions_and_symbols():
    assert CONVENTIONS == ["descriptive", "english", "german", "french", "spanish", "russian"]
    assert GROUP_SYMBOLS == "①②③④⑤⑥⑦⑧"


def test_descriptive_knight_from_white_side():
    assert legal(START, "N-KB3", "descriptive") == {"Nf3"}


def test_descriptive_mirrors_for_black():
    assert legal(after("e4"), "P-K4", "descriptive") == {"e5"}


def test_descriptive_rank_qualifier():
    # Black knights on d7 and h5; "N/2xP" = the knight on Black's 2nd rank (d7) takes the pawn on f6.
    fen = "r1bq1rk1/1p1n1p2/p4Ppp/7n/7Q/3B4/P1P3PP/4RRK1 b - - 0 22"
    assert legal(fen, "N/2xP", "descriptive") == {"Ndxf6"}


def test_descriptive_captured_pawn_file():
    # White bishop d4 can take pawns on e5 only via "BxKP"; a bare "BxP" would also allow others.
    fen = "rnbqkb1r/pppp1ppp/5n2/4p3/3B4/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1"
    assert legal(fen, "BxKP", "descriptive") == {"Bxe5"}


def test_short_pawn_capture():
    assert legal(after("e4", "d5"), "ed", "english") == {"exd5"}


def test_long_algebraic_pawn_is_not_capture():
    assert legal(START, "e2-e4", "english") == {"e4"}


def test_german_letters():
    assert legal(START, "Sf3", "german") == {"Nf3"}


def test_russian_two_letter_king():
    assert legal(after("e4", "e5"), "Кр" + "e2", "russian")  # Russian piece letters, Latin squares == {"Ke2"}


def test_figurine_any_convention():
    assert legal(START, "♘f3", "french") == {"Nf3"}


def test_icon_group_is_any_non_pawn():
    assert legal(START, "①f3", "english") == {"Nf3"}


def test_icon_group_never_pawn():
    assert legal(START, "①e4", "english") == set()


def test_castling_variants():
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    assert legal(fen, "0-0", "english") == {"O-O"}
    assert legal(fen, "O-O", "descriptive") == {"O-O"}
    assert legal(fen, "0-0-0", "german") == {"O-O-O"}
    assert legal(fen, "Castles(Q)", "descriptive") == {"O-O-O"}


def test_check_penalty():
    board = chess.Board()
    nf3 = chess.Move.from_uci("g1f3")
    assert match_cost(board, nf3, read_clues("Nf3+", "english")) == 1.0
    board = chess.Board(after("f3", "e5", "g4"))
    mate = chess.Move.from_uci("d8h4")
    assert match_cost(board, mate, read_clues("Qh4+", "english")) == 0.0
    assert match_cost(board, mate, read_clues("Qh4", "english")) == 0.3


def test_not_a_move():
    assert read_clues("Moscow", "english") is None
    assert read_clues("Sf3", "english") is None
