"""Reasoning-model spike: ask a local thinking model for the moves of each stretch where the decoder is unsure.

One generic task for every kind of trouble: "here is the position, here is what the book prints for the next
moves (OCR, may be damaged), give the moves". Answers are saved for replay; nothing is decided here.

    uv run python -m spike.reason_stretches --split dev           # Capablanca
    uv run python -m spike.reason_stretches --split dev --dry     # print the prompts, ask nothing
    uv run python -m spike.reason_stretches --split dev --game game05

Writes tests/fixtures/books/<book>/reason/<game>.json. Resumable: stretches already answered are skipped.
"""

import argparse
import io
import json
import re
import os
import time
from pathlib import Path

import chess
import chess.pgn
import httpx

import bench.run as run
from bench import reason_answers
from bench.extraction import numbered_entries
from bench.vlm_answers import SavedRowReader, whole_page
from chesspgn import repair
from chesspgn.pipeline import NoGameFound, convert
from chesspgn.tokens import AI_MARK

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
VERSION = 3  # answers from earlier versions of this script were asked from wrong positions and are ignored
MERGE_GAP = 4  # flagged plies this close together are one stretch
MAX_STRETCH = 16  # plies; longer stretches are asked in parts
CONTEXT_ROWS = 2  # printed rows after the stretch, so the answer has to fit what follows

PROMPT = """You are reading a chess game from an old chess book. The text below was read from the page by OCR and \
may be damaged: letters misread, symbols lost, or a move missing entirely. The book may use old descriptive \
notation (P-K4, Kt-KB3, BxKt, R-Q sq, O-O) or algebraic notation.

Position before the first move to find (FEN): {fen}
Moves just before it: {before}
Legal moves now: {legal}

What the book prints for the moves to find, in order:
{slots}

After them the book continues with:
{after}

Where a line shows several printed moves (separated by /), only one is the game move; the others come from the \
book's notes. Give the moves actually played in the game, one for each line above. Write them in standard algebraic \
notation (SAN, e.g. Nf3, exd5, O-O), not in the book's notation; the first must be one of the legal moves listed, \
and each must be legal after the previous ones. Prefer the move that matches the printed text and lets the \
following printed moves be played; where the text is missing or unreadable, choose the move that makes the \
following printed moves possible. Answer with JSON only: \
{{"moves": ["...", "..."]}} with exactly {count} moves."""

SCHEMA = {
    "type": "object",
    "properties": {"moves": {"type": "array", "items": {"type": "string"}}},
    "required": ["moves"],
}


def stretches(flags: list[int], length: int) -> list[tuple[int, int]]:
    plies = sorted(p for p in set(flags) if 1 <= p <= length)
    spans: list[list[int]] = []
    for ply in plies:
        if spans and ply - spans[-1][1] <= MERGE_GAP:
            spans[-1][1] = ply
        else:
            spans.append([ply, ply])
    out = []
    for start, end in spans:
        while end - start + 1 > MAX_STRETCH:
            out.append((start, start + MAX_STRETCH - 1))
            start += MAX_STRETCH
        out.append((start, end))
    return out


def slot_text(entries: dict, board: chess.Board) -> str:
    printed = entries.get((board.fullmove_number, board.turn == chess.BLACK), [])
    return " / ".join(t.replace(AI_MARK, "") for t in printed) or "(nothing printed: missing or unreadable)"


def prompt_for(moves: list[chess.Move], start: int, end: int, entries: dict) -> str:
    board = chess.Board()
    sans = []
    for move in moves[: start - 1]:
        sans.append(board.san(move))
        board.push(move)
    fen, legal = board.fen(), " ".join(sorted(board.san(m) for m in board.legal_moves))
    before = " ".join(sans[-6:]) or "(start of the game)"
    lines, after = [], []
    for ply in range(start, end + 1):
        side = "White" if board.turn == chess.WHITE else "Black"
        lines.append(f"{board.fullmove_number}. {side}: {slot_text(entries, board)}")
        if ply <= len(moves) and board.is_legal(moves[ply - 1]):
            board.push(moves[ply - 1])  # only to step the move numbers along; the decoder's move is not shown
        else:
            board.push(chess.Move.null())
    for _ in range(CONTEXT_ROWS * 2):
        side = "White" if board.turn == chess.WHITE else "Black"
        after.append(f"{board.fullmove_number}. {side}: {slot_text(entries, board)}")
        board.push(chess.Move.null())
    return PROMPT.format(fen=fen, before=before, legal=legal, slots="\n".join(lines), after="\n".join(after),
                         count=end - start + 1)


THINK_TOKENS = 16384  # output budget for thinking + answer
FINISH = ("Here is your reasoning so far:\n\n{thinking}\n\nNow give only the final answer as JSON: "
          '{{"moves": [...]}} with exactly {count} moves in SAN.')


def _chat(model: str, messages: list[dict], think: bool, num_predict: int) -> dict:
    payload = {"model": model, "messages": messages, "think": think, "stream": False,
               "options": {"temperature": 0, "num_ctx": 32768, "num_predict": num_predict}}
    if not think:
        payload["format"] = SCHEMA  # constrained JSON only when not thinking (it can cut the thinking short)
    response = httpx.post(f"{OLLAMA}/api/chat", json=payload, timeout=3600)
    response.raise_for_status()
    return response.json()["message"]


def _moves_from(content: str) -> list[str] | None:
    match = re.search(r"\{.*\}", content or "", re.S)
    if not match:
        return None
    try:
        return [str(m) for m in json.loads(match.group(0)).get("moves", [])]
    except (json.JSONDecodeError, AttributeError):
        return None


def ask(model: str, prompt: str, count: int) -> tuple[list[str], str, float, str]:
    """(moves, error, seconds, thinking). If the thinking uses up the budget before an answer, a short second call
    without thinking turns the reasoning so far into the answer."""
    began = time.monotonic()
    thinking = ""
    try:
        message = _chat(model, [{"role": "user", "content": prompt}], True, THINK_TOKENS)
        thinking = message.get("thinking", "") or ""
        moves = _moves_from(message.get("content", ""))
        if moves is None:
            finish = FINISH.format(thinking=thinking[-12000:], count=count)
            message = _chat(model, [{"role": "user", "content": prompt}, {"role": "user", "content": finish}], False, 1024)
            moves = _moves_from(message.get("content", ""))
        if moves is None:
            return [], "no answer", time.monotonic() - began, thinking
        return moves, "", time.monotonic() - began, thinking
    except Exception as error:  # noqa: BLE001 - a spike: record the failure and go on
        return [], f"{type(error).__name__}: {error}", time.monotonic() - began, thinking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--game")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    repair.row_strip = whole_page
    for manifest in sorted(run.BOOKS.glob("*/book.json")):
        info = json.loads(manifest.read_text(encoding="utf-8"))
        if info["split"] != args.split:
            continue
        book = manifest.parent
        for entry in info["games"]:
            if args.game and entry["id"] != args.game:
                continue
            expected = run.sans((book / "expected" / f"{entry['id']}.pgn").read_text(encoding="utf-8"))
            accept = {int(k): v for k, v in entry.get("accept", {}).items()}
            try:
                games = convert([book / "pages" / p for p in entry["pages"]], run.SavedText(),
                                row_reader=SavedRowReader())
            except NoGameFound:
                continue
            game, _ = run.best_match(expected, games, accept)
            entries = numbered_entries(game.text)
            flagged = [f.ply for f in game.flags if not f.reason.startswith("side:")]
            out_file = book / "reason" / f"{entry['id']}.json"
            saved = json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() else {"stretches": []}
            answered = {(s["start"], s["end"]): s for s in saved["stretches"]
                        if s.get("model") == args.model and not s["error"] and s.get("v") == VERSION}
            current, pins = game, {}
            for start, end in stretches(flagged, len(list(chess.pgn.read_game(io.StringIO(game.pgn)).mainline_moves())) + 1):
                if (start, end) in answered:
                    answer = answered[(start, end)]["moves"]
                else:
                    moves = list(chess.pgn.read_game(io.StringIO(current.pgn)).mainline_moves())
                    if len(moves) < start - 1:
                        continue
                    prompt = prompt_for(moves, start, end, entries)
                    if args.dry:
                        print(f"===== {book.name} {entry['id']} plies {start}-{end}\n{prompt}\n")
                        continue
                    answer, error, seconds, thinking = ask(args.model, prompt, end - start + 1)
                    print(f"{book.name} {entry['id']} plies {start:3}-{end:3}  {seconds:6.0f}s  "
                          f"{' '.join(answer) or error}", flush=True)
                    saved["stretches"] = [s for s in saved["stretches"]
                                          if not (s["start"] == start and s["end"] == end and s.get("model") == args.model)]
                    saved["stretches"].append({"start": start, "end": end, "model": args.model, "moves": answer,
                                               "error": error, "seconds": round(seconds), "prompt": prompt, "v": VERSION,
                                               "thinking": thinking})
                    out_file.parent.mkdir(exist_ok=True)
                    out_file.write_text(json.dumps(saved, indent=1, ensure_ascii=False), encoding="utf-8")
                # the next stretch is asked from the game decoded again with what was accepted so far
                current, _why, _count = reason_answers.step(game, current, pins, start, end, answer, entries)


if __name__ == "__main__":
    main()
