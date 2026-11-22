import chess
import chess.pgn

from chesspgn.variations import build_game
from tests.conftest import expected_moves, fixture, read


def export(game: chess.pgn.Game) -> str:
    return game.accept(chess.pgn.StringExporter(headers=False, columns=None))


def mainline(game: chess.pgn.Game) -> list[chess.Move]:
    return list(game.mainline_moves())


def variation_moves(game: chess.pgn.Game) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    stack: list[tuple[chess.pgn.GameNode, bool]] = [(game, True)]
    while stack:
        node, on_main = stack.pop()
        for i, child in enumerate(node.variations):
            child_main = on_main and i == 0
            if not child_main:
                found.add((node.board().epd(), child.move.uci()))
            stack.append((child, child_main))
    return found


def test_razov_side_lines_match_expected():
    g = build_game(fixture("razov.letters.txt")).game
    assert mainline(g) == expected_moves("razov.pgn")
    assert variation_moves(g) == variation_moves(read(fixture("razov.pgn")))


def test_semicolon_branches_inside_sibling():
    assert "9. Bb3 $2 ( 9. Be2 d5 $10 )" in export(build_game(fixture("razov.letters.txt")).game)


def test_semicolon_branches_on_main_line():
    assert "5... Nf6 ( 5... Na5 $1 )" in export(build_game(fixture("razov.letters.txt")).game)


def test_impossible_book_move_cut_to_comment():
    assert "( 15. Ne4 Nxc3 { 16 Nxd4 Nxd4 17 Qd3" in export(build_game(fixture("zikov.letters.txt")).game)


def test_prose_in_brackets_dropped():
    g = build_game(fixture("stein.ocr.txt")).game
    assert mainline(g) == expected_moves("stein.pgn")
    assert variation_moves(g) == set()


def test_result_header_set():
    assert build_game(fixture("zikov.letters.txt")).game.headers["Result"] == "0-1"


def test_output_parses_clean():
    for name in ("razov.letters.txt", "zikov.letters.txt", "stein.ocr.txt"):
        assert read(str(build_game(fixture(name)).game)).errors == []
