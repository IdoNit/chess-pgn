"""Split page text into games, read headers from the title lines, and name the PGN file."""

import re

import chess

from chesspgn import decoder, notation
from chesspgn.tokens import tokenize
from chesspgn.variations import BuiltGame

MAX_TITLE_LINES = 4
_GAME_START = re.compile(r"(?<![\d.])1\.?\s+(?P<move>\S+)")
_YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")
_PLAYERS = re.compile(r"^(?:Game\s+\d+\s*[:.]\s*)?(?P<white>[^\d\-–—(]+?)\s*[-–—]\s*(?P<black>[^\d\-–—(]+?)\s*(?:\(.*)?$")
_OPENING_WORDS = ("Defence", "Defense", "Opening", "Gambit", "Game", "Attack", "Variation", "System")
_RESULT_OR_END = re.compile(r"\b(1-0|0-1|1:0|0:1|½-½|resign\w*|won|wins|drawn)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(r'[<>:"/\\|?*]')


def split_games(text: str) -> list[str]:
    """One text segment per game: the game's title lines (if any) through to just before the next game."""
    starts = [_header_start(text, p) for p in _game_starts(text)]
    if not starts:
        return []
    starts[0] = 0  # anything before the first game (page prose, tables) stays with it; the decoder skips it
    return [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]


def _game_starts(text: str) -> list[int]:
    depth = _bracket_depth(text)
    board = chess.Board()
    found = []
    for m in _GAME_START.finditer(text):
        if depth[m.start()] == 0 and _is_first_move(m.group("move"), board):
            found.append(m.start())
    return found


def _is_first_move(word: str, board: chess.Board) -> bool:
    tokens = [t for t in tokenize(word) if t.text]
    if not tokens:
        return False
    for convention in notation.CONVENTIONS:
        clues = notation.read_clues(tokens[0].text, convention)
        if clues is not None and any(notation.match_cost(board, m, clues) == 0.0 for m in board.legal_moves):
            return True
    return False


def _bracket_depth(text: str) -> list[int]:
    depth, out = 0, []
    for ch in text:
        if ch in "([{":
            depth += 1
        out.append(depth)
        if ch in ")]}" and depth > 0:
            depth -= 1
    return out


def _header_start(text: str, game_start: int) -> int:
    """Offset of the topmost title line above a game start (up to MAX_TITLE_LINES lines)."""
    line_start = text.rfind("\n", 0, game_start) + 1
    position = line_start
    for _ in range(MAX_TITLE_LINES):
        if position == 0:
            break
        previous_start = text.rfind("\n", 0, position - 1) + 1
        line = text[previous_start:position].strip()
        if line and (_RESULT_OR_END.search(line) or _looks_like_movetext(line)):
            break
        position = previous_start
    return position


def _looks_like_movetext(line: str) -> bool:
    tokens = tokenize(line)
    moves = sum(1 for t in tokens if t.text and any(decoder.looks_like_a_move(t, c) for c in ("english", "descriptive")))
    return moves >= 2


def _title_lines(segment: str) -> list[str]:
    """Non-empty lines directly above the game's first move, nearest first."""
    starts = _game_starts(segment)
    if not starts:
        return []
    above = segment[: starts[0]].splitlines()
    lines = [line.strip() for line in reversed(above) if line.strip()]
    return lines[:MAX_TITLE_LINES]


def extract_headers(segment: str) -> dict[str, str]:
    headers = {"Event": "?", "Site": "?", "Date": "????.??.??", "Round": "?", "White": "?", "Black": "?"}
    opening = None
    for line in _title_lines(segment):
        year = _YEAR.search(line)
        players = _PLAYERS.match(line)
        if headers["White"] == "?" and players is not None and year is None:
            headers["White"], headers["Black"] = players["white"].strip(), players["black"].strip()
        elif headers["Date"].startswith("????") and year is not None:
            headers["Date"] = f"{year.group(1)}.??.??"
            place = (line[: year.start()] + line[year.end():]).strip(" ,.")
            if "," in place:
                event, site = place.rsplit(",", 1)
                headers["Event"], headers["Site"] = event.strip(), site.strip()
            elif place:
                headers["Event"] = place
        elif opening is None and line.rstrip(".").endswith(_OPENING_WORDS):
            opening = line.rstrip(".")
    if opening is not None:
        headers["Opening"] = opening
    return headers


def pgn_filename(headers: dict[str, str]) -> str:
    def known(key: str) -> str:
        value = headers.get(key, "?")
        return "" if value in ("", "?") or value.startswith("????") else _FORBIDDEN.sub("", value).strip()

    name = f"{known('White') or 'White'}-{known('Black') or 'Black'}"
    year = known("Date")[:4] if known("Date") else ""
    tail = " ".join(part for part in (known("Site"), year) if part)
    return f"{name} - {tail}.pgn" if tail else f"{name}.pgn"


def continues_on_next_page(built: BuiltGame) -> bool:
    """No result and the game didn't end in mate: the text probably stops mid-game."""
    return built.decoded.result == "*" and len(built.decoded.moves) > 0
