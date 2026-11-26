"""Where each move of a game is printed: the page, and the box of its row on that page.

From tesseract's lines (text + box) of each page: a move is found by its move number, searching forward through
the pages in reading order, so a note further down that repeats a number cannot take an earlier row. Black's
move is on White's row unless the book prints it on its own ("12. ... R-Q1") before the next row. Used only to
show the reviewer the page and row; nothing is decided from it.
"""

from collections.abc import Callable, Sequence

from chesspgn.repair import _NUMBERED

Box = tuple[int, int, int, int]


def move_places(plies: int, pages: Sequence, lines_of: Callable[[object], list[tuple[str, Box]]],
                sizes: Sequence[tuple[int, int]]) -> list[dict]:
    """For plies 1..plies: {"page": index into pages, "box": [left, top, right, bottom] as fractions of the
    page, or None when the row was not found (then the page is the last one a move was found on)}."""
    found = [[(_number(text), box) for text, box in lines_of(page)] for page in pages]
    cursor = (0, 0)  # (page, line) of the last row placed
    last_page = 0
    places: list[dict] = []
    white_at: tuple[int, int] | None = None
    for ply in range(1, plies + 1):
        number, black = (ply + 1) // 2, ply % 2 == 0
        spot = _black_row(found, white_at, number) if black and white_at else None
        if spot is None:
            spot = _next_row(found, cursor, number, dots=None if black else False)
        if not black:
            white_at = spot
        if spot is None:
            places.append({"page": last_page, "box": None})
            continue
        cursor, last_page = spot, spot[0]
        page, line = spot
        width, height = sizes[page]
        left, top, right, bottom = found[page][line][1]
        places.append({"page": page, "box": [round(left / width, 4), round(top / height, 4),
                                             round(right / width, 4), round(bottom / height, 4)]})
    return places


def _number(text: str) -> tuple[int, bool] | None:
    m = _NUMBERED.match(text)
    return (int(m.group(1)), bool(m["dots"])) if m else None


def _rows_from(found, start: tuple[int, int]):
    page, line = start
    for p in range(page, len(found)):
        for i in range(line if p == page else 0, len(found[p])):
            yield (p, i), found[p][i][0]


def _next_row(found, cursor, number: int, dots: bool | None) -> tuple[int, int] | None:
    """The first row at or after the cursor numbered `number` (with dots or without; None: either)."""
    for spot, numbered in _rows_from(found, cursor):
        if numbered and numbered[0] == number and (dots is None or numbered[1] == dots):
            return spot
    return None


def _black_row(found, white_at, number: int) -> tuple[int, int]:
    """Black's row: a "N. ..." row after White's and before the next numbered row of the game, else White's."""
    for spot, numbered in _rows_from(found, white_at):
        if spot == white_at or not numbered:
            continue
        if numbered == (number, True):
            return spot
        if numbered[0] == number + 1 and not numbered[1]:
            break
    return white_at
