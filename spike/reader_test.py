# /// script
# requires-python = ">=3.10"
# dependencies = ["chess>=1.10", "pillow>=10"]
# ///
"""THROWAWAY spike: which reader keeps the chess piece symbols?

Each reader transcribes each example page; the transcription goes through the board decoder and the
decoded main line is compared with the expected PGN. Usage (Ollama app running):

    uv run spike/reader_test.py
"""

import io
import json
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn
from PIL import Image

SPIKE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SPIKE_DIR / "board_decoder"))  # spike only: decoder lives in a sibling folder

import decoder  # noqa: E402
import ollama_client  # noqa: E402
from ollama_client import ChatSettings  # noqa: E402

EXAMPLES_DIR = SPIKE_DIR.parent / "examples"

TRANSCRIBE = (
    "Transcribe all the text on this chess book page exactly as printed, line by line. "
    "Keep every chess move exactly as printed, including move numbers, brackets and symbols like ! ? + =. "
    "Chess piece figurines MUST be kept: write each one as a capital letter directly before its square "
    "(king K, queen Q, rook R, bishop B, knight N), e.g. a knight figurine followed by f3 is written Nf3. "
    "Do not correct, complete or explain anything. Output only the transcription."
)

READERS: list[dict] = [
    {"name": "glm-ocr", "model": "glm-ocr", "prompt": "Text Recognition:", "scale": 1.0, "think": None},
    {"name": "glm-ocr 2x", "model": "glm-ocr", "prompt": "Text Recognition:", "scale": 2.0, "think": None},
    {"name": "qwen3-vl 8b", "model": "qwen3-vl:8b-instruct", "prompt": TRANSCRIBE, "scale": 1.0, "think": False},
]
_PIECE_BEFORE_SQUARE = re.compile(r"[KQRBN♔♕♖♗♘♚♛♜♝♞](?:x)?[a-h][1-8]")


def main() -> int:
    cases = json.loads((SPIKE_DIR / "cases.json").read_text(encoding="utf-8"))
    out_dir = SPIKE_DIR / "results" / f"readers-{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True)
    rows: list[str] = []
    for case in cases:
        image = EXAMPLES_DIR / case["images"][0]
        expected = list(
            chess.pgn.read_game(io.StringIO((EXAMPLES_DIR / case["expected"]).read_text(encoding="utf-8"))).mainline_moves()
        )
        for reader in READERS:
            print(f"[{case['name']}] {reader['name']} ...", flush=True)
            text, read_seconds = _transcribe(image, reader)
            stem = f"{reader['name'].replace(' ', '_')}__{case['name'].split(' (')[0]}"
            (out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")

            started = time.perf_counter()
            decoded = decoder.decode(text)
            decode_seconds = time.perf_counter() - started
            (out_dir / f"{stem}.pgn").write_text(decoder.to_pgn_movetext(decoded) + "\n", encoding="utf-8")
            rows.append(_row(case["name"], reader["name"], text, decoded, expected, read_seconds, decode_seconds))
            print(f"    read {read_seconds:.0f}s, decode {decode_seconds:.0f}s", flush=True)

    report = (
        "# Reader test\n\nPieces = piece-letter/figurine + square tokens found in the transcription "
        "(a reader that drops figurines scores low).\nMain line = decoded moves matching the expected PGN before the first difference.\n\n"
        "| Game | Reader | Pieces | Main line | Flags | Read | Decode |\n|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report + f"\nOutputs saved in {out_dir}")
    return 0


def _transcribe(image: Path, reader: dict) -> tuple[str, float]:
    settings = ChatSettings(model=reader["model"], cpu_only=False, think=reader["think"])
    if reader["scale"] == 1.0:
        result = ollama_client.chat(settings, reader["prompt"], [image])
        return result.text, result.seconds
    with tempfile.TemporaryDirectory() as tmp:
        scaled = Path(tmp) / "page.png"
        with Image.open(image) as img:
            img.resize((int(img.width * reader["scale"]), int(img.height * reader["scale"])), Image.LANCZOS).save(scaled)
        result = ollama_client.chat(settings, reader["prompt"], [scaled])
        return result.text, result.seconds


def _row(case: str, reader: str, text: str, d: decoder.Decoded, expected: list[chess.Move], read_s: float, decode_s: float) -> str:
    prefix = next((i for i, (a, b) in enumerate(zip(d.moves, expected)) if a != b), min(len(d.moves), len(expected)))
    exact = " ✅" if d.moves == expected else ""
    pieces = len(_PIECE_BEFORE_SQUARE.findall(text))
    return f"| {case} | {reader} | {pieces} | {prefix}/{len(expected)}{exact} | {len(d.repairs)} | {read_s:.0f}s | {decode_s:.0f}s |"


if __name__ == "__main__":
    sys.exit(main())
