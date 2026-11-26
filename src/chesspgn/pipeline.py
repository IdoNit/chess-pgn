"""Screenshots in, checked PGN games out."""

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import chess.pgn

from chesspgn.decoder import Flag
from chesspgn.games import continues_on_next_page, extract_headers, pgn_filename, split_games, title_lines

from chesspgn.icons import annotate, find_icons
from chesspgn.layout import mark_commentary
from chesspgn.reader import Reader
from chesspgn.repair import RowReader, repair_pages
from chesspgn.tokens import normalize_notation
from chesspgn.variations import BuiltGame, build_game

HEADER_ORDER = ["Event", "Site", "Date", "Round", "White", "Black", "Result", "Opening"]


class NoGameFound(Exception):
    """The images contain no recognisable game."""


@dataclass
class ConvertedGame:
    pgn: str
    headers: dict[str, str]
    flags: list[Flag]
    move_count: int
    continues: bool
    filename: str
    text: str  # this game's annotated text, kept for re-decoding with the user's fixes


_TROUBLE = ("unreadable", "missing", "skipped")


def convert(images: list[Path], reader: Reader, row_reader: RowReader | None = None,
            progress: Callable[[str], None] | None = None) -> list[ConvertedGame]:
    """Read the pages and decode their games. With a `row_reader`, rows the board couldn't read get a second
    reading from it (see chesspgn.repair) and the games are decoded again."""
    pages = []
    for n, image in enumerate(images, start=1):
        if progress:
            progress(f"page {n}/{len(images)}")
        pages.append(reader.read(image))
    games = _games(pages, images)
    if row_reader is None:
        return games
    return [_headers_read(g, row_reader) for g in _repaired(games, images, pages, row_reader, progress)]


def _repaired(games, images, pages, row_reader, progress) -> list[ConvertedGame]:
    trouble = {(f.ply + 1) // 2 + d for g in games for f in g.flags if f.reason in _TROUBLE for d in (-1, 0, 1)}
    # any flagged move whose row the OCR lost altogether is worth a look, whatever the flag says
    flagged = {(f.ply + 1) // 2 for g in games for f in g.flags if not f.reason.startswith("side:")}
    if not trouble and not flagged:
        return games
    repaired, rewritten = repair_pages(images, pages, trouble, row_reader,
                                       (lambda done: progress(f"ai {done}")) if progress else None, flagged)
    return _games(repaired, images) if rewritten else games


HEADER_FIELDS = {"white": "White", "black": "Black", "event": "Event", "site": "Site", "opening": "Opening"}


def _headers_read(game: ConvertedGame, helper) -> ConvertedGame:
    """The helper reads the game's heading (players, event, site, year). A value is kept only if the heading
    prints it: the model copies, it does not invent."""
    if not hasattr(helper, "read_headers"):
        return game
    heading = "\n".join(reversed(title_lines(game.text)))
    answer = helper.read_headers(heading) if heading else None
    if not answer:
        return game
    printed = _plain(heading)
    headers = dict(game.headers)
    for key, tag in HEADER_FIELDS.items():
        value = answer.get(key, "").strip(" ,.;:()[]")
        if value and not value.replace(" ", "").isdigit() and _plain(value) in printed:
            headers[tag] = value
    year = answer.get("year", "").strip()
    if re.fullmatch(r"1[5-9]\d\d|20\d\d", year) and year in heading:
        headers["Date"] = f"{year}.??.??"
    if headers == game.headers:
        return game
    pgn = game.pgn
    for tag, value in headers.items():
        line = f'[{tag} "{value}"]'
        if re.search(rf'^\[{tag} "', pgn, flags=re.M):
            pgn = re.sub(rf'^\[{tag} "[^"\n]*"\]$', lambda _m: line, pgn, count=1, flags=re.M)
        else:  # a tag the game had no line for (Opening): after the last tag line
            last = list(re.finditer(r'^\[\w+ "[^"\n]*"\]$', pgn, flags=re.M))[-1]
            pgn = pgn[: last.end()] + "\n" + line + pgn[last.end():]
    return replace(game, pgn=pgn, headers=headers, filename=pgn_filename(headers))


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.lower()).split())


def _games(pages: list[str], images: list[Path]) -> list[ConvertedGame]:
    text = normalize_notation("\n".join(pages))
    if not uses_piece_letters(text):  # figurines lose their symbols in OCR; letters don't
        text = annotate(text, find_icons(images))
    games = [game_from_text(segment) for segment in split_games(mark_commentary(text))]
    games = [g for g in games if g.move_count > 0]
    if not games:
        raise NoGameFound("No chess game found in the images")
    return games


_LETTER_MOVE = re.compile(r"(?<![A-Za-z])[KQRBN][a-h]?[1-8]?[x×:]?[a-h][1-8]|\b[KQRBNP]-[QK]?[RNB]?[1-8]")
MIN_LETTER_MOVES = 5


def uses_piece_letters(text: str) -> bool:
    """Whether the page prints pieces as letters (Nc3, N×d5, N-KB3): then there are no figurines to find."""
    return len(_LETTER_MOVE.findall(text)) >= MIN_LETTER_MOVES


def game_from_text(
    segment: str,
    pins: dict[int, chess.Move] | None = None,
    headers: dict[str, str] | None = None,
    end: int | None = None,
) -> ConvertedGame:
    built = build_game(segment, pins, end=end)
    found = extract_headers(segment) | (headers or {})
    found["Result"] = built.decoded.result
    return ConvertedGame(
        pgn=render_pgn(built, found),
        headers=found,
        flags=built.flags,
        move_count=len(built.decoded.moves),
        continues=continues_on_next_page(built),
        filename=pgn_filename(found),
        text=segment,
    )


def render_pgn(built: BuiltGame, headers: dict[str, str]) -> str:
    game = built.game
    for key in list(game.headers):
        del game.headers[key]
    for key in HEADER_ORDER:
        if key in headers:
            game.headers[key] = headers[key]
    return str(game) + "\n"
