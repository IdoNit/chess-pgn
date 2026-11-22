from chesspgn.games import continues_on_next_page, extract_headers, pgn_filename, split_games
from chesspgn.variations import build_game
from tests.conftest import fixture


def test_two_games_split():
    games = split_games(fixture("razov.letters.txt") + "\n" + fixture("zikov.letters.txt"))
    assert len(games) == 2
    assert "Game 2: Zikov-VK" in games[1] and "Zikov" not in games[0]


def test_single_game_not_split():
    assert len(split_games(fixture("stein.ocr.txt"))) == 1


def test_table_and_prose_are_not_games():
    assert len(split_games(fixture("razov.ocr.txt"))) == 1


def test_headers_razov():
    h = extract_headers(split_games(fixture("razov.ocr.txt"))[0])
    assert (h["White"], h["Black"], h["Event"], h["Date"], h["Opening"]) == (
        "VK", "Razov", "Leningrad Junior Ch", "1946.??.??", "Hungarian Defence",
    )


def test_headers_stein():
    h = extract_headers(split_games(fixture("stein.ocr.txt"))[0])
    assert (h["White"], h["Black"], h["Event"], h["Site"], h["Date"], h["Round"]) == (
        "Stein", "Korchnoi", "Zonal", "Moscow", "1964.??.??", "?",
    )


def test_filename():
    assert pgn_filename({"White": "Stein", "Black": "Korchnoi", "Site": "Moscow", "Date": "1964.??.??"}) == "Stein-Korchnoi - Moscow 1964.pgn"


def test_filename_unknowns_omitted():
    assert pgn_filename({"White": "Zikov", "Black": "VK", "Site": "?", "Date": "????.??.??"}) == "Zikov-VK.pgn"


def test_filename_windows_safe():
    assert pgn_filename({"White": 'A/B', "Black": "C?", "Site": "?", "Date": "????.??.??"}) == "AB-C.pgn"


def test_continues_on_next_page():
    text = fixture("stein.ocr.txt")
    assert not continues_on_next_page(build_game(text))
    assert continues_on_next_page(build_game(text[: text.index("31 RxB")]))
