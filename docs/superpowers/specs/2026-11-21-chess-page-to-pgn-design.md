# Chess page → PGN converter — Design

Date: 2026-11-21 · Status: **draft, pending owner answers (see end)**

## Intent

Ido's father converts chess-book pages into PGN files. He takes a screenshot of the page and asks ChatGPT for the PGN. This works, but the free ChatGPT plan limits how much he can do. The goal is a free tool with no usage limits that runs on his Windows PC. It takes the same screenshots and returns a clean `.pgn` file, which he then moves to his tablet (Chess PGN Master, ChessX, etc.).

Reference input/output pair: `Stein-Korchnoi_Moscow_1964.jpg` → `Stein-Korchnoi_Moscow_1964.pgn` (in this folder).

## Goals

- Input: one or more screenshots (PNG/JPG) of **one complete game, starting at move 1**. Several images are read in order, for games that run across pages.
- Output: one valid `.pgn` file containing the moves **only**: no comments, no annotation symbols (`!`, `?`, `!!`), no variations. The reference output is the target format.
- Notations: English **descriptive** (`P-K4`, `N/2xP`, `QR-K1`, `BxKP`) and **algebraic** (SAN, long algebraic, figurine `♘f3`).
- Free and unlimited: runs fully locally, with no API keys and no network after install.
- Never hand back a silently wrong game. Every move is replayed on a board and must be legal.

## Non-goals

- Reading diagrams (the father confirmed they aren't needed).
- Fragments, puzzles or studies that start from a position other than move 1.
- Keeping the book's comments or variations.
- Non-English piece letters (German, Russian, ...). Possible later.
- Mobile or tablet app, cloud hosting, multi-user use.

## Anti-goals (failure even if it "works")

- A PGN that opens but contains a wrong move. Every move must come from the page and be legal on the board.
- A broken or corrupted PGN file (the father's main complaint about ChatGPT when it adds notes).
- Setup or usage the father can't do alone after the one-time install.

## Constraints

- Father's machine: Windows PC. GPU unknown, so the design assumes **CPU only**. Up to a few minutes per page is acceptable.
- Ido develops on a Mac (Apple Silicon); the father's PC is x86-64, so images must be built for `linux/amd64`.
- Ido installs Docker Desktop on the father's PC once.

## Approaches considered

1. **Docker + local web page + small open OCR model** — **chosen.** It runs the same on the Mac and the PC, it's easy to update, and Docker Desktop is a one-time install. Cost: an image several GB in size.
2. Native Windows installer, no Docker. Lighter for the father, but hard to build and test from a Mac, and fragile.
3. Tesseract-only, no AI model. Tiny and fast, but less accurate on varied fonts and scans. Kept as a candidate in the engine test (below) rather than as the architecture.

## Architecture

```
screenshots ──► [1] OCR model ──► raw text ──► [2] cleaner ──► move tokens ──► [3] replayer (python-chess) ──► PGN
                                                                                    │
                                                                                    └─► problems (move #, token) ──► UI
```

1. **Reader (OCR).** A small open model transcribes the page text verbatim. The engine is chosen by the test in the next section. Candidates: GLM-OCR (~0.9B), PaddleOCR-VL, Tesseract. The interface is `read(image) -> str`, so the engine can be swapped.
2. **Cleaner.** Removes everything that isn't the game's main line:
   - parenthesized text (book comments, which often contain *analysis moves* like "Threatening 24 RxN…" and must be dropped before parsing),
   - quoted remarks, annotation symbols (`!`, `?`, `!!`, `!?`, `?!`),
   - the title block, diagram residue, and the "and Black resigned" text.
   - It then splits what's left into move-number markers and move tokens. It must tolerate missing or broken move numbers (the reference page has "24 BxKP P-R3 BxN PxN", with no "25"). Move order comes from alternating White and Black moves, not from the printed numbers.
3. **Replayer.** For each token, it builds the set of legal moves in the current position and keeps those that match the token:
   - Algebraic: parse SAN or long algebraic, and map figurines to letters.
   - Descriptive: squares are read from the moving side's point of view (K-file = e, QB = c, …, ranks counted from the mover's side). Qualifiers narrow the match: piece side (`QR`, `KN`), captured piece and file (`BxKP`), rank or file of origin (`N/2xP`), castling `0-0` / `0-0-0`, check `+`, and promotion.
   - Exactly one match: play it. Zero or several: stop and report the move number, the side and the raw token.
4. **Result and headers.** The result comes from the text when present (`1-0`, "resigned", "and wins"). Otherwise the field is editable in the UI, defaulting to `*`. See open question 2 for the other headers.

## User interface (local web page)

The father opens a desktop shortcut, and the browser opens `http://localhost:<port>`.

```
┌──────────────────────────────────────────────┐
│  Drop screenshots here (in page order)       │
│  [img1] [img2]                     [Convert] │
├──────────────────────────────────────────────┤
│  White [Stein, Leonid ]  Black [Korchnoi, V.]│
│  Event [USSR Zonal]  Site [Moscow]  Year[1964]│
│  Result [1-0]                                │
├──────────────────────────────────────────────┤
│  ✅ 93 moves read — all legal                 │
│  [Download Stein-Korchnoi_Moscow_1964.pgn]   │
└──────────────────────────────────────────────┘
```

When a move fails, the page shows "Move 31 (White): couldn't read `RxB!!`" next to the editable transcribed text. He fixes the text and clicks Convert again; the image isn't re-read. *(My call.)*

The file name follows his convention: `White-Black_Site_Year.pgn`. *(My call.)*

## Engine test (first step of implementation)

Throwaway benchmark on CPU, `linux/amd64` emulation acceptable for accuracy, native timing noted separately:

- Measure: exact move-list match against the reference PGN, and seconds per page.
- Pick the most accurate engine that runs in under about 3 minutes per page on CPU. If they're tied, pick the smallest.
- Needs more sample pages than the single reference (see open question 1).

## Packaging and delivery

- One Docker image containing the Python app, its dependencies and the **model weights baked in**, so it works offline. *(My call.)*
- `docker-compose.yml` with a restart policy, plus `Start Chess PGN.bat` on the father's desktop. The script starts the container if needed and opens the browser. Docker Desktop is set to start at login. *(My call.)*
- The image is built for `linux/amd64` with `docker buildx` and published to GHCR from a GitHub repo. Updates are `docker compose pull`. *(My call.)*

## Engineering notes for the builder

- Python 3.12. Code follows the `dignified-python` standards (modern typing, pathlib, explicit checks).
- Web layer: FastAPI with a single static page, or Gradio. Builder's choice; keep it simple.
- Tests: golden tests on the reference pair plus any extra samples, and unit tests for descriptive parsing (qualifiers, ambiguity, castling, promotion, missing move numbers).
- The git repo is initialized as part of the plan; this doc moves to `docs/superpowers/specs/` and is committed then.

## Left to the builder

The web framework, the exact OCR prompt and settings, the cleaner's heuristics, and the image base.

## Open questions for the owner (answers change what gets built)

1. **More sample pages.** Can the father provide 3–5 more screenshots with the PGN he considers correct? Ideally they'd come from different books: at least one in algebraic notation, one from a DjVu, and one game that runs across two pages. With only one example, the engine choice and the parser are tuned to one book.
2. **Headers the page doesn't show.** His reference PGN has `Date 1964.03.02`, `Round 10` and `ECO B84`, which aren't on the page. ChatGPT filled them in from memory. A local tool can't know the exact date or round. Options:
   - (a) Leave unknown fields as `?`, following the PGN standard.
   - (b) Also compute `ECO`/`Opening` from the moves with a free openings table. This is reliable.

   My pick is (b) plus editable fields in the UI. Is that OK for him, or does he rely on the date and round being filled in?
