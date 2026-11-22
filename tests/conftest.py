import io
from pathlib import Path

import chess
import chess.pgn

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def fixture_path(name: str) -> Path:
    return FIXTURES / name


def read(pgn: str) -> chess.pgn.Game:
    return chess.pgn.read_game(io.StringIO(pgn))


def expected_moves(pgn_name: str) -> list[chess.Move]:
    return list(read(fixture(pgn_name)).mainline_moves())

