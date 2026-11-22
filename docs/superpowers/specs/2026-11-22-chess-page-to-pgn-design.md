# Chess page → PGN — Design

Date: 2026-11-22 · Status: **draft for owner approval** · Replaces the 2026-11-21 draft.

## 1. Intent

Ido's father converts chess-book pages into PGN files. He takes a screenshot of a page (from PDF, DjVu or
other e-books) and today asks ChatGPT to convert it. The free plan limits him, and ChatGPT sometimes returns
corrupted files or wrong moves without saying so. The goal is a **free, unlimited, local** tool on his
Windows PC that turns the same screenshots into a correct `.pgn` file. He then copies the file to his tablet
(Chess PGN Master, ChessX, ...).

The core promise: **every move in the output is legal and comes from the page. Anything the tool had to guess
is shown to him, never hidden.**

## 2. Goals

- Input: one or more screenshots (PNG/JPG), in page order. They may hold **one game or several** (e.g. the end
  of one game and the start of the next). The tool finds every game, converts each, and lets him tick which to
  keep. Each kept game becomes its own entry in "My games", downloadable separately or as one combined PGN.
- Output: a valid PGN with
  - the main line,
  - side lines from brackets as **real PGN variations**,
  - the book's `!` `?` `!!` `??` `!?` `?!` as NAGs and `=` as `$10`,
  - the result,
  - headers taken from the page.
  - The book's prose commentary is dropped.
  - A side line containing an impossible move is cut there, and the rest of it is kept as a `{comment}`.
- Notation detected **automatically**, with no per-book setup:
  - English descriptive (`P-K4`, `N/2xP`, `QR-K1`)
  - algebraic: short, long, abbreviated pawn captures (`ed`)
  - piece letters in English, German, French, Spanish and Russian
  - **figurines**, including books whose figurines the text reader cannot read
- Fixes OCR damage on its own (wrong characters, lost characters, junk tokens, missing move numbers or moves), and
  **flags** every place it repaired.
- After converting: one-click download, plus an optional review screen to check and fix moves on a board.
- History ("My games") so he can reopen, fix and re-download a past game without re-uploading.
- Runs fully offline after install, on modest hardware (no GPU assumed).

## 3. Non-goals

- Reading diagrams (the father confirmed they aren't needed).
- Games or fragments that start from a position other than move 1 (puzzles, studies).
- Keeping prose commentary.
- Mobile or tablet app, cloud hosting, accounts, multiple users.

## 4. Anti-goals (failure even if it "works")

- A file that opens fine but contains a wrong move **with no flag on it**.
- A broken or unreadable PGN file.
- Anything the father can't do alone after the one-time install, or anything that needs configuration per book.

## 5. Constraints

- Father's machine: Windows PC with unknown specs. The design assumes a **CPU only** and modest RAM, so
  only small models. Up to a couple of minutes per page is fine.
- Ido develops on a Mac (Apple Silicon); images are built for `linux/amd64`.
- Ido installs Docker Desktop on the father's PC once.

## 6. Approaches considered (with evidence from the spike, `spike/`)

| Approach | Result on the 3 example pages |
|---|---|
| A. OCR model → local LLM (Qwen3-VL 8B) writes the PGN | Wrong games: the LLM invented moves and got stuck in repetition loops (e.g. "Qe2 Qc7 Qe3 Qc6…") even from a *perfect* transcription. 3/93, 2/53, 1/40 correct moves. |
| B. Vision LLM (Qwen3-VL 8B) straight from image to PGN | The same failures. 3/93, 4/53, 0/40. |
| **C. OCR model reads text → board decoder (code) turns text into moves** | **Exact on all three main lines** (93/93, 53/53, 40/40), plus 26/28 and all correctly placed side lines. |

**C is chosen.** Small local LLMs can't keep track of a chess position. The board can: at any moment only about
30 moves are legal, and a whole game is a long chain of legality constraints that corrects the OCR.

Readers tested for C: GLM-OCR (0.9B) read text well (the descriptive page word for word, about 20 s per page on a
Mac GPU). **Every reader, including Qwen3-VL when told to keep them, dropped figurine symbols.** Figurines
are therefore recovered from the image directly (§7.2).

## 7. Architecture

```
screenshots ─┬─► [1] Text reader (GLM-OCR via Ollama) ──────────────┐
             └─► [2] Figurine finder (Tesseract + template matching) ┤
                                                                     ▼
                                     [3] Linker (icons → moves in text)
                                                                     ▼
                                     [4] Board decoder (python-chess)  ──► PGN + flags
                                                                     ▼
                                     [5] Web app (review, history, download)
```

### 7.1 Text reader
GLM-OCR through a local Ollama server (prompt `Text Recognition:`). It's called once per image, and the texts
are joined in page order. Its interface is `read(image) -> str`, so the reader can be swapped.

### 7.2 Figurine finder (no AI, no per-book setup)
1. Tesseract character boxes: a character sitting right before a square (`Af3`, `&xg7`) marks an icon position.
   The letters Tesseract guesses for icons are unreliable, but their positions are good.
2. Crop those icons and group them by shape (normalized correlation).
3. Every group with 2 or more members becomes a template. Template matching over the whole page finds all the
   remaining icons.
4. For each icon, read the move text to its right (Tesseract, whitelist `a-h1-8x`) and record its reading
   order (Tesseract's block, paragraph and line).

The groups have no names: ①②③… The decoder learns which piece each group is (§7.4). On non-figurine
pages this step finds nothing and changes nothing. When a game spans several screenshots, the groups and
templates are built from all of them together, so ① means the same piece on every page. *(Ido's call.)*

Spike result: on both figurine pages every icon in the text was found, with no false detections in the diagrams
or tables. A piece type appearing only once (a king) gets no template and is handled by the decoder's fallback.

### 7.3 Linker
An order-preserving alignment (dynamic programming) between the moves in the reader's text and the icon
sequence. Costs: same text 0; same square with a different prefix (`d4`/`xd4`) 0.5; misread square 1;
text move with no icon 0.3 (pawn moves); icon with no text move 1. The matched moves get the group symbol
(`①f3`). Linking is allowed to be imperfect: the decoder treats group symbols as strong evidence, not as facts.

### 7.4 Board decoder
- **Tokenize** the text: move numbers, moves, `!?` annotations, brackets, `;`, results, and prose.
- **Game start:** the first `1` followed by a token that reads as a legal first move. Tables and prose before it
  are ignored. **Several games:** every such start that follows a result (`1-0`, "resigned", ...) or a new title
  block begins a new game; each segment is decoded on its own. A game cut off at the end of the text is kept
  and marked "continues on next page?".
- **Notation:** every convention is tried, and the one whose game stays legal wins. (Spike: correct in 8 of 8
  styles on 3 games.)
- **Reading a token = clues, not parsing.** Piece, destination (descriptive squares from the mover's side),
  from-file or rank, capture and captured piece, check, mate, promotion. Each legal move is tested against
  the clues.
- **Search:** a beam search over the whole game for the cheapest legal game. Options per token:
  - read it exactly,
  - read it after an OCR correction (confusable characters, a lost or extra character, a dropped `x`),
  - skip it as prose,
  - replace an unreadable token with any legal move,
  - insert a move missing from the text.

  Move numbers and `+`/`x`/`mate` are soft checksums. Costs are in `decoder.py`.
- **Icon groups:** each search state carries a group→piece mapping that it learns as it goes. Two groups may be
  the same piece (split groups happen). Contradicting a learned group costs extra, and dropping a misplaced
  group symbol is the cheapest correction.
- **Side lines:** bracket groups are parsed as a tree, with `;` separating alternatives. Each alternative is
  decoded from every candidate line at its move number, in a strict mode: OCR corrections are allowed, but a
  move-like token is never skipped. A line stops at the first impossible move, and the rest becomes a
  `{comment}`. Ties between legal placements are broken by "the replaced move is by the same piece" and then
  "same piece type". Spike: the tricky `[8...Ne5! 9 Bb3? …; 9 Be2 d5 =]` and `[5 c3 …; 5...Na5!]` were both
  placed correctly.
- **Result:** from `1-0` / `1:0` / "Black resigned" / "and won" (the side that moved last), or from checkmate.
- **Pinned moves:** the decoder accepts "ply N is move X" constraints (fixes from the review screen) and
  searches around them.
- **Flags:** every correction (OCR repair, ignored icon, invented or missing move, number mismatch) is
  reported with the move and the reason. Where the best and second-best games are close in cost, the first
  move where they differ is flagged too.

Robustness measured in the spike (Stein page, OCR-style damage injected): 3 errors → 10/10 exact;
6 → 10/10; 15 → 8/10 exact and 2/10 flagged. **0 silent errors.**

### 7.5 Headers
- White, Black and Event/Site/Year are taken from the title lines above the game, by heuristic (e.g. "Game 1:
  VK-Razov / Leningrad Junior Ch 1946"); the opening name is taken from the page when present.
- Unknown values follow the PGN standard (`Date "1946.??.??"`, `Round "?"`). This matches what his ChatGPT
  files already do.
- All fields are editable in the app.

### 7.6 Web app
A single local page served by the container. He opens it from a desktop shortcut.

**Language: Hebrew, right-to-left.** All labels, buttons and messages are in Hebrew (`dir="rtl"`). Chess
content lives in left-to-right islands (`dir="ltr"` / `<bdi>`): the move list, board, suggested moves, player
names, PGN text and filenames. Directions are never mixed inside one line. Approved from a mockup on
2026-11-22. UI strings are kept in one file so an English version can be added later.

**Convert screen:** drop one or more screenshots in page order, then Convert. The result card shows:
- players, event and number of moves,
- either "✅ all checked" or "⚠ N moves to check",
- **Download PGN** (always available) and **Review / edit**.

**Review screen:**
- The screenshot on one side, zoomable. The move list and a board on the other.
- Flagged moves are highlighted, and the screen opens on the first one.
- Clicking any move shows the position before it and what the book printed. He fixes a move by picking one of
  the suggested alternatives or by dragging a piece (legal moves only).
- A fix is **pinned**, and the decoder runs again around it, so the rest of the game re-aligns.
- Header fields are editable.

**My games:** a list of past conversions, newest first, each with its flag status. He can open, edit and
re-download any of them.

**Storage:** `Documents\Chess PGN\<date> <White>-<Black>\` holds the screenshots, `game.pgn` and a
`state.json` (fixes and flags). It's an ordinary folder he can browse and back up.
*(Ido's call.)* The PGN filename follows his convention: `White-Black - Site Year.pgn`.

### 7.7 Packaging
- `docker-compose.yml` runs two services:
  - `ollama`, with the `glm-ocr` model baked into the image so it works offline,
  - `app`, with Python, Tesseract and the web UI.
- The father's `Documents\Chess PGN` folder is mounted into the app.
- A `Start Chess PGN.bat` desktop shortcut runs `docker compose up -d` and opens the browser.
- Docker Desktop is set to start at login.
- Images are built for `linux/amd64` and published to GHCR from the project repo. Updates are
  `docker compose pull`.

## 8. Testing

- **Golden pages:** the 3 example pages (`examples/`). Their expected main lines must match exactly.
  Side lines are compared move by move. His ChatGPT files differ in how they store side lines, so those
  are reviewed by eye.
- **Held-out acceptance set:** `examples/holdout/` (3 pages from Ido). **Not opened during design or
  development.** They're run once on the finished build to measure it honestly.
- **Unit tests:** notation clue reading for every convention; descriptive squares from both sides; castling and
  promotion; `ed`-style captures.
- **Robustness tests:** the spike's synthetic notation renderings and OCR-damage injection, run with fixed seeds
  as regression tests. The target is **zero silent errors**.
- **Speed:** each page must finish within about 2 minutes on CPU-only Docker. Measured on Ido's Mac with Docker
  limited to 4 CPUs.

## 9. Known limits and planned improvements (left to the builder, in order)

1. **Speed:** the decoder tries every notation over the whole game (10–50 s per page in a 2-CPU sandbox).
   It should commit to the notation after the first ~10 moves.
2. **Chess-engine tiebreak** (Stockfish, offline) for flagged spots where two legal moves both fit, e.g. a
   side line's `♗a4` read as pawn `a4` after its icon was linked to a neighbouring move.
3. Linking around two-column layouts (text beside a diagram) shifts some icons by one move. The decoder
   recovers, and each case is flagged; a better reading order would reduce the flags.
4. A piece type with a single icon on the page has no template; the decoder's insertion repair covers it.

## 10. Left to the builder

The web framework (FastAPI + one static page is suggested), the board widget (any free JS chessboard plus
chess.js), the base images, the exact header heuristics, and the code layout. Python code follows the
`dignified-python` standards. The spike code in `spike/board_decoder/` is the reference implementation of
§7.2–7.4. It should be rebuilt cleanly, with tests, not copied as is.
