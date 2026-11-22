"""Screenshots in, checked PGN games out."""

from dataclasses import dataclass
from pathlib import Path

import chess.pgn

from chesspgn.decoder import Flag
from chesspgn.games import continues_on_next_page, extract_headers, pgn_filename, split_games
from chesspgn.icons import annotate, find_icons
from chesspgn.reader import Reader
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


def convert(images: list[Path], reader: Reader) -> list[ConvertedGame]:
    text = "\n".join(reader.read(image) for image in images)
    annotated = annotate(text, find_icons(images))
    games = [game_from_text(segment) for segment in split_games(annotated)]
    games = [g for g in games if g.move_count > 0]
    if not games:
        raise NoGameFound("No chess game found in the images")
    return games


def game_from_text(segment: str, pins: dict[int, chess.Move] | None = None, headers: dict[str, str] | None = None) -> ConvertedGame:
    built = build_game(segment, pins)
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
