"""Local web app: convert screenshots, review and fix moves, history, download."""

import io
import os
import tempfile
from pathlib import Path
from urllib.parse import quote

import chess
import chess.pgn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from chesspgn.decoder import suggest_moves
from chesspgn.pipeline import NoGameFound, convert, game_from_text
from chesspgn.reader import OllamaReader, Reader, ReaderUnavailable
from chesspgn.storage import GameNotFound, GameStore

STATIC = Path(__file__).parent / "static"
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def create_app(store_root: Path, reader: Reader) -> FastAPI:
    app = FastAPI(title="Chess PGN")
    store = GameStore(store_root)

    @app.exception_handler(GameNotFound)
    def _not_found(_request, _exc):
        return JSONResponse({"error": "not_found"}, status_code=404)

    @app.post("/api/convert")
    def convert_images(images: list[UploadFile]):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, upload in enumerate(images, start=1):
                suffix = Path(upload.filename or "").suffix.lower()
                path = Path(tmp) / f"page-{i:02d}{suffix if suffix in ALLOWED_SUFFIXES else '.png'}"
                path.write_bytes(upload.file.read())
                paths.append(path)
            try:
                games = convert(paths, reader)
            except NoGameFound:
                return JSONResponse({"error": "no_game"}, status_code=422)
            except ReaderUnavailable:
                return JSONResponse({"error": "engine_down"}, status_code=503)
            records = [store.save(game, paths) for game in games]
        return {
            "games": [
                {
                    "id": r.id,
                    "title": r.title,
                    "filename": r.filename,
                    "flags_open": r.flags_open,
                    "move_count": g.move_count,
                    "continues": g.continues,
                }
                for r, g in zip(records, games)
            ]
        }

    @app.get("/api/games")
    def list_games():
        return [
            {"id": r.id, "title": r.title, "created": r.created, "flags_open": r.flags_open, "filename": r.filename}
            for r in store.list()
        ]

    @app.get("/api/games/{game_id}")
    def get_game(game_id: str):
        return _detail(store, game_id)

    @app.delete("/api/games/{game_id}")
    def delete_game(game_id: str):
        store.delete(game_id)
        return {"deleted": game_id}

    @app.get("/api/games/{game_id}/images/{name}")
    def get_image(game_id: str, name: str):
        return FileResponse(store.image(game_id, name))

    @app.post("/api/games/{game_id}/pin")
    def pin(game_id: str, body: dict):
        _record, state = store.load(game_id)
        move = _legal_pin(state["pgn"], body)
        if move is None:
            return JSONResponse({"error": "illegal_pin"}, status_code=422)
        ply, uci = move
        _redecode(store, game_id, state, {**state["pins"], str(ply): uci}, state["headers"])
        return _detail(store, game_id)

    @app.post("/api/games/{game_id}/dismiss")
    def dismiss(game_id: str, body: dict):
        """The user checked a flagged move and confirms it as is."""
        _record, state = store.load(game_id)
        state["flags"] = [f for f in state["flags"] if not (f["ply"] == body.get("ply") and f["reason"] == body.get("reason"))]
        store.update(game_id, state)
        return _detail(store, game_id)

    @app.get("/api/download")
    def download_several(ids: str):
        pgns = [store.load(game_id)[1]["pgn"].strip() for game_id in ids.split(",") if game_id]
        disposition = f"attachment; filename*=UTF-8''{quote('games.pgn')}"
        return Response("\n\n".join(pgns) + "\n", media_type="text/plain; charset=utf-8", headers={"Content-Disposition": disposition})

    @app.put("/api/games/{game_id}/headers")
    def set_headers(game_id: str, body: dict):
        _record, state = store.load(game_id)
        headers = {**state["headers"], **{str(k): str(v) for k, v in body.items() if k != "Result"}}
        _redecode(store, game_id, state, state["pins"], headers)
        return _detail(store, game_id)

    @app.get("/api/games/{game_id}/download")
    def download(game_id: str):
        _record, state = store.load(game_id)
        disposition = f"attachment; filename*=UTF-8''{quote(state['filename'])}"
        return Response(state["pgn"], media_type="text/plain; charset=utf-8", headers={"Content-Disposition": disposition})

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def app_from_env() -> FastAPI:
    """Entry point for `uvicorn --factory chesspgn.web.app:app_from_env`."""
    data = Path(os.environ.get("CHESSPGN_DATA", "/data"))
    fake = os.environ.get("CHESSPGN_FAKE_READER")
    reader: Reader = _FixtureReader(Path(fake)) if fake else OllamaReader(os.environ.get("OLLAMA_URL", "http://ollama:11434"))
    return create_app(data, reader)


class _FixtureReader:
    """Development only: answers with tests/fixtures/<n>.ocr.txt in turn (stein, razov, zikov)."""

    def __init__(self, folder: Path) -> None:
        self.texts = [folder / f"{n}.ocr.txt" for n in ("stein", "razov", "zikov")]
        self.calls = 0

    def read(self, image: Path) -> str:
        text = self.texts[self.calls % len(self.texts)].read_text(encoding="utf-8")
        self.calls += 1
        return text


def _legal_pin(pgn: str, body: dict) -> tuple[int, str] | None:
    """(ply, uci) if `body` names a legal move at that ply of the current main line, else None."""
    ply, uci = body.get("ply"), body.get("uci")
    if not isinstance(ply, int) or not isinstance(uci, str):
        return None
    game = chess.pgn.read_game(io.StringIO(pgn))
    moves = list(game.mainline_moves())
    if not 1 <= ply <= len(moves) + 1:
        return None
    board = game.board()
    for move in moves[: ply - 1]:
        board.push(move)
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return None
    return (ply, uci) if board.is_legal(move) else None


def _redecode(store: GameStore, game_id: str, state: dict, pins: dict[str, str], headers: dict[str, str]) -> None:
    game = game_from_text(store.text(game_id), {int(p): chess.Move.from_uci(u) for p, u in pins.items()}, headers)
    played = list(chess.pgn.read_game(io.StringIO(game.pgn)).mainline_moves())
    honoured = {p: u for p, u in pins.items() if int(p) <= len(played) and played[int(p) - 1].uci() == u}
    flags = [
        {"ply": f.ply, "token": f.token, "chosen": f.chosen, "reason": f.reason}
        for f in game.flags
        if f.reason.startswith("side:") or str(f.ply) not in honoured
    ]
    # A pin the new decode could not play is dropped and reported, never silently ignored.
    flags += [{"ply": int(p), "token": u, "chosen": "", "reason": "pin_dropped"} for p, u in pins.items() if p not in honoured]
    state.update(
        pgn=game.pgn,
        headers=game.headers,
        flags=sorted(flags, key=lambda f: f["ply"]),
        pins=honoured,
        filename=game.filename,
        move_count=game.move_count,
        continues=game.continues,
    )
    store.update(game_id, state)


def _detail(store: GameStore, game_id: str) -> dict:
    record, state = store.load(game_id)
    game = chess.pgn.read_game(io.StringIO(state["pgn"]))
    moves, boards = [], []
    board = game.board()
    for ply, move in enumerate(game.mainline_moves(), start=1):
        boards.append(board.copy())
        moves.append({"ply": ply, "san": board.san(move), "uci": move.uci(), "fen_before": board.fen()})
        board.push(move)
    flags = []
    for f in state["flags"]:
        suggestions: list[str] = []
        if not f["reason"].startswith("side:") and 1 <= f["ply"] <= len(moves):
            before = boards[f["ply"] - 1]
            suggestions = suggest_moves(before, f["token"], chess.Move.from_uci(moves[f["ply"] - 1]["uci"]))
        flags.append({**f, "suggestions": suggestions})
    return {
        "id": record.id,
        "title": record.title,
        "filename": state["filename"],
        "pgn": state["pgn"],
        "headers": state["headers"],
        "flags": flags,
        "pins": state["pins"],
        "moves": moves,
        "move_count": state["move_count"],
        "continues": state["continues"],
        "images": [f"/api/games/{record.id}/images/{name}" for name in state["image_names"]],
    }
