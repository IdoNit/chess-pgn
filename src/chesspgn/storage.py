"""Converted games, one ordinary folder each, under the father's Documents/Chess PGN."""

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from chesspgn.pipeline import ConvertedGame

_UNSAFE = re.compile(r'[<>:"/\\|?*]')


class GameNotFound(Exception):
    pass


@dataclass
class GameRecord:
    id: str  # the folder name
    folder: Path
    title: str
    created: str
    flags_open: int
    filename: str


class GameStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, game: ConvertedGame, images: list[Path]) -> GameRecord:
        self.root.mkdir(parents=True, exist_ok=True)
        folder = self._new_folder(game.headers)
        for image in images:
            shutil.copy(image, folder / image.name)
        (folder / "text.txt").write_text(game.text, encoding="utf-8")
        state = {
            "pgn": game.pgn,
            "headers": game.headers,
            "flags": [asdict(f) for f in game.flags],
            "pins": {},
            "image_names": [image.name for image in images],
            "created": datetime.now().isoformat(timespec="microseconds"),
            "filename": game.filename,
            "move_count": game.move_count,
            "continues": game.continues,
        }
        return self._write(folder, state)

    def list(self) -> list[GameRecord]:
        if not self.root.exists():
            return []
        records = [self._record(folder) for folder in self.root.iterdir() if (folder / "state.json").exists()]
        return sorted(records, key=lambda r: r.created, reverse=True)

    def load(self, game_id: str) -> tuple[GameRecord, dict]:
        folder = self._folder(game_id)
        return self._record(folder), self._state(folder)

    def update(self, game_id: str, state: dict) -> GameRecord:
        return self._write(self._folder(game_id), state)

    def delete(self, game_id: str) -> None:
        shutil.rmtree(self._folder(game_id))

    def text(self, game_id: str) -> str:
        return (self._folder(game_id) / "text.txt").read_text(encoding="utf-8")

    def image(self, game_id: str, name: str) -> Path:
        folder = self._folder(game_id)
        path = folder / name
        if path.parent != folder or name not in self._state(folder)["image_names"]:
            raise GameNotFound(name)
        return path

    # ------------------------------------------------------------ internals

    def _folder(self, game_id: str) -> Path:
        folder = self.root / game_id
        if folder.resolve().parent != self.root.resolve() or not (folder / "state.json").exists():
            raise GameNotFound(game_id)
        return folder

    def _new_folder(self, headers: dict[str, str]) -> Path:
        players = _UNSAFE.sub("", f"{headers.get('White', '?')}-{headers.get('Black', '?')}")
        players = players[:100].rstrip(". ").strip() or "game"  # Windows can't open names ending in "." or a space
        base = f"{datetime.now():%Y-%m-%d} {players}"
        folder, n = self.root / base, 1
        while folder.exists():
            n += 1
            folder = self.root / f"{base} ({n})"
        folder.mkdir()
        return folder

    def _write(self, folder: Path, state: dict) -> GameRecord:
        (folder / "game.pgn").write_text(state["pgn"], encoding="utf-8")
        (folder / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._record(folder)

    @staticmethod
    def _state(folder: Path) -> dict:
        return json.loads((folder / "state.json").read_text(encoding="utf-8"))

    def _record(self, folder: Path) -> GameRecord:
        state = self._state(folder)
        return GameRecord(
            id=folder.name,
            folder=folder,
            title=state["filename"].removesuffix(".pgn"),
            created=state["created"],
            flags_open=len(state["flags"]),
            filename=state["filename"],
        )
