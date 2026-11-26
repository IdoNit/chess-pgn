"""Where each move is printed: the page, and the box of its row (fractions of the page), from tesseract's lines."""

from chesspgn.move_places import move_places

W, H = 1000, 2000
PAGES = [
    [("GAME 1. RUY LOPEZ", (100, 50, 900, 80)),
     ("1. P-K4 P-K4", (200, 100, 800, 130)),
     ("2. Kt-KB3 Kt-QB3", (200, 140, 800, 170))],
    [("3. B-Kt5", (200, 100, 500, 130)),
     ("A comment that mentions 3 B-B4 as well.", (100, 140, 900, 170)),
     ("3. ... P-QR3", (200, 180, 800, 210)),
     ("4. B-R4 Kt-B3", (200, 220, 800, 250))],
]


def lines(page):
    return PAGES[page]


def test_each_move_is_placed_on_its_page_and_row():
    places = move_places(8, [0, 1], lines, [(W, H), (W, H)])
    assert [p["page"] for p in places] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert places[0]["box"] == places[1]["box"] == [0.2, 0.05, 0.8, 0.065]  # row 1, both moves
    assert places[4]["box"] == [0.2, 0.05, 0.5, 0.065]  # "3. B-Kt5"
    assert places[5]["box"] == [0.2, 0.09, 0.8, 0.105]  # Black's 3rd move, printed after the comment
    assert places[6]["box"] == places[7]["box"] == [0.2, 0.11, 0.8, 0.125]


def test_a_move_whose_row_is_not_found_stays_on_the_page_before_without_a_box():
    places = move_places(10, [0, 1], lines, [(W, H), (W, H)])
    assert places[8] == {"page": 1, "box": None} and places[9] == {"page": 1, "box": None}


def test_the_search_goes_forward_so_a_later_note_cannot_take_an_earlier_row():
    pages = [[("1. P-K4 P-K4", (200, 100, 800, 130)), ("2. Kt-KB3 Kt-QB3", (200, 140, 800, 170)),
              ("1 P-Q4, P-Q4; the other first move", (100, 300, 900, 330))]]
    places = move_places(4, [0], lambda p: pages[p], [(W, H)])
    assert places[0]["box"] == [0.2, 0.05, 0.8, 0.065]
