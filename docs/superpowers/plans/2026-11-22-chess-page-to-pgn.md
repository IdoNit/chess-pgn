# Chess Page → PGN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local, free, offline Windows tool (Docker) that turns chess-book page screenshots into correct PGN files. Every guess it makes is flagged, and the father can review and fix moves in a Hebrew web UI.

**Architecture:** GLM-OCR (through Ollama) reads the page text. Tesseract and template matching find figurine icons and group them by shape. A board decoder (python-chess beam search) turns the noisy text into the cheapest legal game, including side lines, and learns which icon group is which piece. A FastAPI app serves a Hebrew RTL page for converting, reviewing and history.

**Tech Stack:**
- Python 3.12, uv, python-chess, opencv-python-headless, numpy, pytesseract (Tesseract 5)
- FastAPI and uvicorn, httpx (Ollama client)
- Vanilla JS with vendored `chess.js` and `cm-chessboard`
- Docker Compose (`ollama/ollama` + app), pytest

**Spec:** `docs/superpowers/specs/2026-11-22-chess-page-to-pgn-design.md`. Reference implementation of spec §7.2–7.4: `spike/board_decoder/` (rebuild cleanly with tests; don't copy as is).

## Global Constraints

- Python ≥ 3.12, managed with uv. Code follows the `dignified-python` standards: LBYL, pathlib, explicit `encoding="utf-8"`, absolute imports, no re-exports.
- Runs fully offline after install. No network calls except to the local Ollama server. JS libraries are vendored, not taken from a CDN.
- No per-book configuration anywhere. Notation and icon groups are detected automatically.
- **Every correction the decoder makes is a flag.** A wrong move without a flag is a failing bug.
- Output PGN must parse with `chess.pgn.read_game` with `game.errors == []`.
- Unknown headers follow PGN style: `Date "1946.??.??"`, `Round "?"`.
- PGN filename: `White-Black - Site Year.pgn` (e.g. `Stein-Korchnoi - Moscow 1964.pgn`). Unknown parts are omitted, never written as `?`.
- Storage root: `Documents\Chess PGN`, mounted into the container at `/data`. One folder per game: `YYYY-MM-DD White-Black[ (n)]/`, containing the images, `game.pgn` and `state.json`.
- UI: Hebrew, `dir="rtl"`. Chess content (moves, board, suggestions, names, PGN, filenames) sits in `dir="ltr"` / `<bdi>` islands, and directions are never mixed inside one line. All UI strings live in `src/chesspgn/web/static/strings.he.json`.
- Speed: ≤ 120 s per page, CPU only, in Docker limited to 4 CPUs.
- Images: `linux/amd64`. Text reader: Ollama model `glm-ocr`, prompt `Text Recognition:`.
- `examples/holdout/` is **never opened** until Task 13.

## Review Focus

1. **A screenshot with no chess moves** (wrong page, blank, only prose): the user expects a clear Hebrew message ("לא נמצא משחק בתמונה"), not a crash and not an empty PGN. Owned by Task 8.
2. **A game split across two screenshots mid-move** (page 1 ends "8...", page 2 starts "Nxd4?"): one game, the move joined correctly. Owned by Task 8.
3. **Ollama not running or the model missing:** a Hebrew message saying the engine isn't running, not a stack trace or a hang. Owned by Task 8 (reader) and Task 10 (HTTP 503).
4. **A badly damaged or very long page:** a result within the time limit, with flags, never an endless spinner. Owned by Task 4 (time budget).
5. **The same game converted twice:** a second folder `… (2)`, nothing overwritten. Owned by Task 9.

---

## File structure

```
pyproject.toml                      uv project, deps, pytest config
src/chesspgn/
  notation.py      clue reading + move matching (all conventions, icon groups)
  tokens.py        tokenizer + OCR-confusion variants
  decoder.py       beam search: main line, strict side-line mode, pins, flags, time budget
  variations.py    bracket tree, side-line placement, build chess.pgn.Game
  games.py         split text into games, header heuristics, PGN filename
  icons.py         figurine finder (Tesseract + templates) + linker to text
  reader.py        Ollama GLM-OCR client
  pipeline.py      convert(images) -> list[ConvertedGame]
  storage.py       GameStore (Documents/Chess PGN)
  web/app.py       FastAPI routes
  web/static/      index.html, app.js, app.css, strings.he.json, vendor/
tests/
  fixtures/        example images, expected PGNs, saved GLM-OCR texts
  test_*.py        one per module
docker/
  Dockerfile.app, Dockerfile.ollama, docker-compose.yml, Start Chess PGN.bat
```

---

### Task 1: Project setup + notation clue reading

**Files:**
- Create: `pyproject.toml`, `.gitignore` (`.venv`, `__pycache__`, `spike/results`, `.DS_Store`), `src/chesspgn/__init__.py` (empty), `src/chesspgn/notation.py`
- Create: `tests/fixtures/` (copy the `examples/*.jpg` and `*.pgn` files, and the saved GLM-OCR texts from `spike/results/20261122-143610/A__*.ocr.txt` renamed `stein.ocr.txt`, `razov.ocr.txt`, `zikov.ocr.txt`; images renamed `stein.jpg`, `razov.jpg`, `zikov.jpg`; PGNs `stein.pgn`, `razov.pgn`, `zikov.pgn`)
- Test: `tests/test_notation.py`

**Interfaces:**
- Produces:
  - `CONVENTIONS: list[str] = ["descriptive", "english", "german", "french", "spanish", "russian"]`
  - `GROUP_SYMBOLS = "①②③④⑤⑥⑦⑧"`
  - `@dataclass(frozen=True) class Clues` with the fields from `spike/board_decoder/notation.py`, including `piece_class: int | None`
  - `read_clues(token: str, convention: str) -> Clues | None`
  - `match_cost(board: chess.Board, move: chess.Move, clues: Clues) -> float | None`: `None` = contradicts, else a penalty (missing `+` → 0.3, `+` on a non-check → 1.0, `mate` not mate → 1.0, unstated promotion → 0.5)

- [ ] **Step 1:** `git init`, the uv project (`uv init --package --name chesspgn`, `uv add chess opencv-python-headless numpy pytesseract fastapi uvicorn httpx python-multipart`, `uv add --dev pytest`), the fixtures copied, and a first commit of the spec, the spike and the fixtures.
- [ ] **Step 2: Write failing tests** in `tests/test_notation.py`. A helper `legal(fen, token, convention) -> set[str]` returns the SAN of the legal moves with `match_cost(...) is not None`.

```python
START = chess.STARTING_FEN
def test_descriptive_knight_from_white_side(): assert legal(START, "N-KB3", "descriptive") == {"Nf3"}
def test_descriptive_mirrors_for_black():
    b = chess.Board(); b.push_san("e4")
    assert legal(b.fen(), "P-K4", "descriptive") == {"e5"}
def test_descriptive_rank_qualifier(): ...  # "N/2xP" picks the knight standing on the mover's 2nd rank
def test_descriptive_captured_pawn_file(): ... # "BxKP" only matches capturing the e-file pawn
def test_short_pawn_capture():  # position with exd5 legal
    assert legal("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "ed", "english") == {"exd5"}
def test_long_algebraic_pawn_is_not_capture(): assert legal(START, "e2-e4", "english") == {"e4"}
def test_german_letters(): assert legal(START, "Sf3", "german") == {"Nf3"}
def test_russian_two_letter_king(): ...  # "Кре2" after 1.e4 e5 -> {"Ke2"}
def test_figurine_any_convention(): assert legal(START, "♘f3", "french") == {"Nf3"}
def test_icon_group_is_any_non_pawn(): assert legal(START, "①f3", "english") == {"Nf3"}
def test_icon_group_never_pawn(): assert legal(START, "①e4", "english") == set()
def test_castling_variants(): ...  # "0-0", "O-O", "0-0-0", "Castles(Q)" each match the right castling
def test_check_penalty(): ...      # "Nf3+" on a non-check move -> cost 1.0; correct "+" -> 0.0
```

- [ ] **Step 3: Run** `uv run pytest tests/test_notation.py -v`. Expected: FAIL (module not found).
- [ ] **Step 4: Implement** `notation.py`, following `spike/board_decoder/notation.py`: clue regexes, `_leading_piece`, descriptive file names `["QR","QN","QB","Q","K","KB","KN","KR"]`, and bishop side by square colour.
- [ ] **Step 5: Run.** Expected: all PASS. **Commit** `feat: notation clue reading and matching`.

### Task 2: Tokenizer and OCR variants

**Files:** Create `src/chesspgn/tokens.py`. Test `tests/test_tokens.py`.

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Token(text: str, annotation: str = "", number: int | None = None, black_to_move: bool = False)`
  - `tokenize(text: str) -> list[Token]`
  - `variants(text: str) -> tuple[tuple[str, float], ...]`: the token itself at cost 0, then corrections

  Costs:
  - `COST_SUBSTITUTION = 1.0`
  - `COST_DELETION = 1.5`
  - `COST_INSERTION = 1.2` (single layer, alphabet `KQRBNP12345678abcdefgh` plus `x`)
  - `COST_DROP_GROUP = 0.8` (a leading group symbol)

  Confusion pairs as in the spike `_CONFUSIONS`.

- [ ] **Step 1: Write failing tests:**

```python
def test_glued_number(): assert tokenize("8...Nxd4?") == [Token("", number=8, black_to_move=True), Token("Nxd4", "?")]
def test_spaced_dots(): assert [t.number for t in tokenize("16 . . . N-Q2")] == [16, None]
def test_dashes_normalised(): assert tokenize("B–K2")[0].text == "B-K2"
def test_result_kept(): assert tokenize("27 Bc5! 1:0")[-1].text == "1:0"
def test_variants_include_x_insertion(): assert ("PxP", 1.0) in variants("PP")
def test_variants_drop_group_first(): assert variants("②c4")[1] == ("c4", 0.8)
def test_variants_bounded(): assert len(variants("QR-KB1+")) < 2500
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement**, following the spike's `tokenize`, `_variants`, `_one_edit` and `_insertions`. `variants` is `lru_cache`d.
- [ ] **Step 4: Run.** Expected: PASS. **Commit** `feat: tokenizer and OCR variants`.

### Task 3: Main-line decoder

**Files:** Create `src/chesspgn/decoder.py`. Test `tests/test_decoder.py`.

**Interfaces:**
- Consumes: `notation.read_clues`, `notation.match_cost`, `notation.CONVENTIONS`, `tokens.Token`, `tokens.variants`.
- Produces:
  - `@dataclass(frozen=True) class Flag(ply: int, token: str, chosen: str, reason: str)`, where `ply` is 1-based within its line and `chosen` is SAN.
  - `@dataclass class Decoded(convention: str, moves: list[chess.Move], annotations: list[str], token_index: list[int], cost: float, flags: list[Flag], result: str, mapping: tuple[int, ...])`
  - `decode_tokens(tokens: list[Token], convention: str | None = None, pins: dict[int, chess.Move] | None = None) -> Decoded`
  - `game_start(tokens: list[Token], convention: str) -> int`
  - `looks_like_a_move(token: Token, convention: str) -> bool`

Costs (copy them exactly):
- skip a move-like token: 3.0; skip prose: 0
- wildcard: 4.0
- number mismatch: 1.5
- icon-group conflict: 2.0

Beam width: 100. A missing move is inserted only for the top 5 states.

Flag reasons, used verbatim (Task 12 maps them to Hebrew):
- `"ocr"` (cost ≥ 1.0)
- `"icon"` (dropped group symbol)
- `"unreadable"` (wildcard)
- `"missing"` (inserted move)
- `"close_call"`: the cheapest alternative game is within 1.5 of the best one's cost; flag the first ply where they differ.

- [ ] **Step 1: Write failing tests**, using fixtures through a `fixture(name) -> str` helper in `tests/conftest.py`. Compare main lines with `expected_moves(pgn_name) -> list[chess.Move]`.

```python
def test_stein_real_ocr_exact():
    d = decode_tokens(tokenize(fixture("stein.ocr.txt")))
    assert d.convention == "descriptive" and d.moves == expected_moves("stein.pgn") and d.result == "1-0"
def test_game_start_skips_table(): ...  # razov.ocr.txt: tokens before index of first "1" "e4" ignored
def test_result_from_and_won(): ...    # zikov text -> "0-1"
def test_qxp_capital_x_repaired_and_flagged():
    text = fixture("stein.ocr.txt").replace("47 QxP+", "47 QXP+")
    d = decode_tokens(tokenize(text))
    assert d.moves == expected_moves("stein.pgn") and any(f.ply == 93 and f.reason == "ocr" for f in d.flags)
def test_icon_groups_learned(): ...    # synthetic: Razov main line rendered with ①=N ②=B ③=Q ④=R -> exact, mapping learns N,B,Q,R
def test_pins_override():              # pin ply 3 to a different legal move -> that move is used and later plies re-align
    ...
@pytest.mark.parametrize("n_errors,seed", [(3, 3), (6, 6), (15, 15)])
def test_damage_never_silent(n_errors, seed): ...  # port spike stress.corrupt; result is exact OR has a flag at/before first wrong ply
@pytest.mark.parametrize("style", ["english","short_captures","long","german","french","russian","figurine"])
def test_notation_styles_exact(style): ...          # port spike stress.render on razov.pgn main line
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** the search, following spike `_search` / `_expand` / `_reading_match` / `_offer_missing_move`:
  - State key: `(board._transposition_key(), len(moves), mapping)`.
  - Pins: at ply `p` in `pins`, only the pinned move is offered.
  - `close_call` is computed in `_finish` from the beam.
- [ ] **Step 4: Run.** Expected: PASS. **Commit** `feat: board decoder main line`.

### Task 4: Speed and time budget

**Files:** Modify `src/chesspgn/decoder.py`. Test `tests/test_decoder_speed.py`.

**Interfaces:**
- Produces: `decode_tokens(..., time_budget_s: float = 90.0)`. When the budget runs out, the remaining tokens go through beam width 10 and no missing-move inserts, and a `Flag(ply, "", "", "time_budget")` is added at the first ply decoded that way.

- [ ] **Step 1: Write failing tests:**

```python
def test_convention_committed_early(monkeypatch):
    calls = count_calls(monkeypatch, decoder, "_decode_with_convention")
    decode_tokens(tokenize(fixture("stein.ocr.txt")))
    assert calls.full_length_runs == 1          # only the winning convention decodes past the probe
def test_stein_under_15s(): assert timed(decode_tokens, tokenize(fixture("stein.ocr.txt"))) < 15
def test_time_budget_returns_flagged(): ...     # time_budget_s=0.01 -> returns a legal game with a "time_budget" flag
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** `_decode_with_convention(tokens, convention, start, stop=None) -> Decoded` (the per-convention search) and the notation probe: decode the first 24 tokens from game start under every convention, keep the cheapest, and decode the full text only with that one. If the full decode's cost per move is over 1.0, retry the second-best convention and keep the cheaper result.
- [ ] **Step 4: Run.** Expected: PASS. **Commit** `perf: early notation commit and time budget`.

### Task 5: Side lines

**Files:** Create `src/chesspgn/variations.py`. Modify `decoder.py`: add `decode_line`. Test `tests/test_variations.py`.

**Interfaces:**
- Consumes: `decoder.decode_tokens`, `decoder.game_start`, `decoder.looks_like_a_move`, `Flag`.
- Produces:
  - `decoder.decode_line(tokens: list[Token], convention: str, board: chess.Board, mapping: tuple[int, ...]) -> Line`, where `Line(moves, annotations, token_index, cost, cut_at: int | None, flags: list[Flag])`. It is strict: a move-like token is never skipped, there are no wildcards or inserts, and decoding stops at the first unmatched move.
  - `@dataclass class BuiltGame(game: chess.pgn.Game, decoded: Decoded, flags: list[Flag])`. `flags` holds main-line flags plus side-line flags, each side-line flag with `reason` prefixed `"side:"`.
  - `build_game(text: str, pins: dict[int, chess.Move] | None = None) -> BuiltGame`

Placement tie-breaks, exactly: same from-square −0.6, same piece type −0.3, recent sibling −0.05, first move equal to the replaced move +5.0. NAGs: `! ? !! ?? !? ?!` map to 1–6, and `=` maps to 10.

- [ ] **Step 1: Write failing tests**, with the hand-typed pages copied from `spike/board_decoder/testpages/` into `tests/fixtures/razov.letters.txt` and `zikov.letters.txt`:

```python
def test_razov_side_lines_match_expected():
    g = build_game(fixture("razov.letters.txt")).game
    assert mainline(g) == expected_moves("razov.pgn")
    assert variation_moves(g) == variation_moves(read("razov.pgn"))   # all 28, none extra
def test_semicolon_branches_inside_sibling(): ...   # "9. Bb3 $2 ( 9. Be2 d5 $10 )" present in exported PGN
def test_semicolon_branches_on_main_line(): ...     # "5... Nf6 ( 5... Na5 $1 )"
def test_impossible_book_move_cut_to_comment():
    pgn = export(build_game(fixture("zikov.letters.txt")).game)
    assert "( 15. Ne4 Nxc3 { 16 Nxd4 Nxd4 17 Qd3" in pgn
def test_prose_in_brackets_dropped(): ...           # stein.ocr.txt -> no variations, main line exact
def test_output_parses_clean(): ...                 # read_game(export(g)).errors == [] for all three fixtures
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement**, following spike `variations.py` (`parse_tree`, `_attach_group`, `_place`). The cut comment keeps the move number before the cut token.
- [ ] **Step 4: Run.** Expected: PASS. **Commit** `feat: side lines`.

### Task 6: Several games, headers, filename

**Files:** Create `src/chesspgn/games.py`. Test `tests/test_games.py`.

**Interfaces:**
- Produces:
  - `split_games(text: str) -> list[str]`: a new segment starts at a `1` followed by a legal first move, when it comes after a result token, a "resigned"/"won" phrase, or a title line (`Game \d+`), or when it isn't the first game.
  - `extract_headers(segment: str) -> dict[str, str]` with keys `White, Black, Event, Site, Date, Round, Opening`.
  - `pgn_filename(headers: dict[str, str]) -> str`
  - `continues_on_next_page(built: BuiltGame) -> bool`: true when there's no result and the text ends right after a move.

- [ ] **Step 1: Write failing tests:**

```python
def test_two_games_split(): assert len(split_games(fixture("razov.letters.txt") + "\n" + fixture("zikov.letters.txt"))) == 2
def test_single_game_not_split(): assert len(split_games(fixture("stein.ocr.txt"))) == 1
def test_headers_razov():
    h = extract_headers(split_games(fixture("razov.ocr.txt"))[0])
    assert (h["White"], h["Black"], h["Event"], h["Date"], h["Opening"]) == ("VK", "Razov", "Leningrad Junior Ch", "1946.??.??", "Hungarian Defence")
def test_headers_stein(): ...   # White "Stein", Black "Korchnoi", Site "Moscow", Date "1964.??.??", Round "?"
def test_filename(): assert pgn_filename({"White": "Stein", "Black": "Korchnoi", "Site": "Moscow", "Date": "1964.??.??"}) == "Stein-Korchnoi - Moscow 1964.pgn"
def test_filename_unknowns_omitted(): assert pgn_filename({"White": "Zikov", "Black": "VK", "Site": "?", "Date": "????.??.??"}) == "Zikov-VK.pgn"
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Headers come from the ≤4 non-move lines above the game start:
  - a line containing `X-Y` or `X–Y` (after an optional `Game n:` prefix) gives White and Black;
  - a line with a 4-digit year gives Event, Site and Date. A comma splits Event from Site ("Zonal, Moscow 1964" → Event "Zonal", Site "Moscow"); otherwise the whole line minus the year is the Event;
  - a line ending in "Defence", "Defense", "Opening", "Gambit", "Game" or "Attack" gives Opening.
- [ ] **Step 4: Run.** Expected: PASS. **Commit** `feat: game splitting and headers`.

### Task 7: Figurine finder and linker

**Files:** Create `src/chesspgn/icons.py`. Test `tests/test_icons.py`. Needs Tesseract: `brew install tesseract` on the Mac; it's in the Docker image.

**Interfaces:**
- Consumes: `notation.GROUP_SYMBOLS`.
- Produces:
  - `@dataclass(frozen=True) class IconHit(image_index: int, box: tuple[int, int, int, int], group: int, square_text: str | None, order: tuple[int, ...])`
  - `find_icons(images: list[Path]) -> list[IconHit]`: templates are built from all images together; the hits are in reading order (image, Tesseract block/par/line, x).
  - `annotate(text: str, hits: list[IconHit]) -> str`: the order-preserving DP alignment with costs exactly: same text 0, same square 0.5, misread 1.0, text move with no icon 0.3, icon with no text 1.0.

- [ ] **Step 1: Write failing tests:**

```python
def test_razov_icons_found_and_grouped():
    hits = find_icons([fixture_path("razov.jpg")])
    assert len(hits) >= 45 and 4 <= len({h.group for h in hits}) <= 7
def test_no_icons_on_descriptive_page(): assert find_icons([fixture_path("stein.jpg")]) == []
def test_no_hits_inside_diagram():   # razov.jpg diagram occupies x 830-1280, y 1480-1940 (original pixels)
    hits = find_icons([fixture_path("razov.jpg")])
    assert not any(830 < h.box[0] < 1280 and 1480 < h.box[1] < 1940 for h in hits)
def test_annotate_keeps_pawn_moves(): assert "4 d4 ed" in annotate(fixture("razov.ocr.txt"), find_icons([fixture_path("razov.jpg")]))
def test_end_to_end_razov_main_exact():
    text = annotate(fixture("razov.ocr.txt"), find_icons([fixture_path("razov.jpg")]))
    assert mainline(build_game(text).game) == expected_moves("razov.pgn")
def test_end_to_end_zikov_main_exact(): ...  # same for zikov
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement**, following spike `icons.py`:
  - seed icons: Tesseract char boxes before a square;
  - grouping: greedy correlation ≥ 0.75 on 24×24 normalized crops;
  - templates: groups with ≥ 2 members;
  - matching: `TM_CCOEFF_NORMED` ≥ 0.72, overlap (IoU) suppression at 0.3;
  - square reading: Tesseract `--psm 7` with whitelist `abcdefghx12345678`.
- [ ] **Step 4: Run.** Expected: PASS. **Commit** `feat: figurine finder and linker`.

### Task 8: Reader and pipeline

**Files:** Create `src/chesspgn/reader.py` and `src/chesspgn/pipeline.py`. Test `tests/test_pipeline.py`.

**Interfaces:**
- Produces:
  - `class ReaderUnavailable(Exception)`
  - `class Reader(Protocol): def read(self, image: Path) -> str`
  - `OllamaReader(base_url: str = "http://ollama:11434", model: str = "glm-ocr", timeout_s: float = 300).read(image) -> str`, which raises `ReaderUnavailable` on a connection error, HTTP 404 (model missing) or an empty answer
  - `@dataclass class ConvertedGame(pgn: str, headers: dict[str, str], flags: list[Flag], move_count: int, continues: bool, filename: str)`
  - `class NoGameFound(Exception)`
  - `convert(images: list[Path], reader: Reader) -> list[ConvertedGame]`: read each image, join the texts with `"\n"`, `annotate` with `find_icons(images)`, then `split_games` → `build_game` → headers → PGN string

- [ ] **Step 1: Write failing tests**, using a `FakeReader` that returns the fixture texts:

```python
def test_convert_stein():
    [g] = convert([fixture_path("stein.jpg")], FakeReader({"stein.jpg": "stein.ocr.txt"}))
    assert g.filename == "Stein-Korchnoi - Moscow 1964.pgn" and read(g.pgn).errors == []
def test_no_game_raises():   # Review Focus 1
    with pytest.raises(NoGameFound):
        convert([fixture_path("stein.jpg")], FakeReader(returns="Chapter 3. Some prose only."))
def test_split_across_pages():   # Review Focus 2
    page1, page2 = split_at(fixture("stein.ocr.txt"), "23...P-K4!")   # page 1 ends "...23 N–N5", page 2 starts "23...P-K4!"
    [g] = convert([p1, p2], FakeReader(...))
    assert mainline(read(g.pgn)) == expected_moves("stein.pgn")
def test_ollama_down():   # Review Focus 3
    with pytest.raises(ReaderUnavailable):
        OllamaReader("http://127.0.0.1:9").read(fixture_path("stein.jpg"))
@pytest.mark.live
def test_live_glm_ocr_razov(): ...   # only with `pytest -m live` and Ollama running: real reader -> main line exact
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** `OllamaReader` posts to `/api/chat` with `{"model", "messages":[{"role":"user","content":"Text Recognition:","images":[b64]}], "stream": false, "options":{"temperature":0}}`. Register the `live` marker in `pyproject.toml` and skip it by default.
- [ ] **Step 4: Run.** Expected: PASS. **Commit** `feat: reader and conversion pipeline`.

### Task 9: Game storage

**Files:** Create `src/chesspgn/storage.py`. Test `tests/test_storage.py`.

**Interfaces:**
- Produces:
  - `@dataclass class GameRecord(id: str, folder: Path, title: str, created: str, flags_open: int, filename: str)`
  - `GameStore(root: Path)` with:
    - `.save(game: ConvertedGame, images: list[Path]) -> GameRecord`
    - `.list() -> list[GameRecord]` (newest first)
    - `.load(id: str) -> tuple[GameRecord, dict]`, where the dict is the `state.json` contents: `{"pgn", "headers", "flags", "pins": {ply: uci}, "image_names"}`
    - `.update(id: str, state: dict) -> GameRecord`, which rewrites `game.pgn` and `state.json`

  The record id is the folder name.

- [ ] **Step 1: Write failing tests** using `tmp_path`:

```python
def test_save_layout(tmp_path): r = GameStore(tmp_path).save(stein_game(), [img]); assert (r.folder / "game.pgn").exists() and (r.folder / "stein.jpg").exists()
def test_same_game_twice_gets_suffix(tmp_path):   # Review Focus 5
    s = GameStore(tmp_path); a = s.save(g, [img]); b = s.save(g, [img])
    assert a.folder != b.folder and b.folder.name.endswith("(2)") and (a.folder / "game.pgn").read_text(encoding="utf-8") == g.pgn
def test_list_newest_first(tmp_path): ...
def test_update_rewrites_pgn(tmp_path): ...
```

- [ ] **Step 2–4:** Run (FAIL), implement, run (PASS). **Commit** `feat: game storage`.

### Task 10: Web API

**Files:** Create `src/chesspgn/web/app.py`. Test `tests/test_api.py` (FastAPI `TestClient`, `FakeReader` injected through `app.state.reader`).

**Interfaces:**
- Consumes: `convert`, `GameStore`, `build_game` (for re-decoding with pins), `ReaderUnavailable`, `NoGameFound`.
- Produces (JSON):
  - `POST /api/convert`, multipart `images[]` in order → `{"games": [{"id", "title", "move_count", "flags_open", "continues", "filename"}]}`. Status 422 `{"error": "no_game"}` when no game is found; 503 `{"error": "engine_down"}` when Ollama is unreachable.
  - `GET /api/games` → a list of records.
  - `GET /api/games/{id}` → `{"pgn", "headers", "flags", "pins", "images": [urls]}`. `GET /api/games/{id}/images/{name}` returns the image.
  - `POST /api/games/{id}/pin`, body `{"ply": int, "uci": str}` → re-decodes with all the pins and returns the same shape as GET.
  - `PUT /api/games/{id}/headers`, body `{header: value}` → the same shape.
  - `GET /api/games/{id}/download` → `text/plain` PGN with `Content-Disposition` set to `filename`.
  - `GET /` → `static/index.html`. Static files are served from `/static`.

  The app factory is `create_app(store_root: Path, reader: Reader) -> FastAPI`; the module-level `app = create_app(Path("/data"), OllamaReader())`.

- [ ] **Step 1: Write failing tests:** convert → list → get → pin → download round trip on Stein; a no-game image gives 422 `no_game`; an `ErrorReader` raising `ReaderUnavailable` gives 503 `engine_down`; `POST /pin {"ply": 49, "uci": "d3g6"}` (White's 25th move, Bxg6 instead of Bxf6) makes ply 49 equal `d3g6` in the returned PGN, the PGN parses clean, and `pins == {"49": "d3g6"}`.
- [ ] **Step 2–4:** Run (FAIL), implement, run (PASS). Re-decoding with pins re-runs `build_game` on the stored text: save the joined, annotated text as `text.txt` in the game folder (add it to Task 9's `save` and test it). **Commit** `feat: web API`.

### Task 11: Hebrew web UI

**Files:** Create `src/chesspgn/web/static/index.html`, `app.js`, `app.css`, `strings.he.json`, and `vendor/` (`chess.js` 1.x ESM, `cm-chessboard` 8.x with its assets, downloaded once and committed). Test `tests/test_ui_strings.py`, plus a manual checklist.

**Interfaces:**
- Consumes: the API from Task 10.
- Produces three views in one page, switched by hash (`#/`, `#/game/{id}`, `#/games`), matching the approved mockup:
  1. **Convert:** a drop zone that accepts multiple files, kept in the order dropped and reorderable by drag. Then a result card per game: Latin text in `<bdi>`; "⚠ N מסעים לבדיקה" or "✅ כל המסעים נבדקו"; primary "הורדת PGN" and secondary "בדיקה ועריכה". Each card has a "שמור" checkbox when there are several games. A card with `continues` shows "ממשיך בעמוד הבא?".
  2. **Review:** the screenshot (zoomable with the mouse wheel and drag), an LTR move list with flagged moves highlighted, and a `cm-chessboard` showing the position before the selected move. The panel shows the flag reason, what the book printed, up to 3 alternative legal moves as buttons, and "או הזיזו כלי על הלוח". Drag-to-move is validated with chess.js. A choice calls `/pin`, re-renders, and moves to the next open flag. The header fields are editable.
  3. **My games:** a list with title, date, flag status and "פתיחה".

- [ ] **Step 1: Write failing test** `test_ui_strings.py`:
  - every `data-i18n` key used in `index.html` and `app.js` exists in `strings.he.json`;
  - every flag reason (`ocr`, `icon`, `unreadable`, `missing`, `close_call`, `time_budget` and their `side:` forms) has a string;
  - the errors `no_game` and `engine_down` have strings: "לא נמצא משחק בתמונה" and "מנוע הקריאה לא פועל – הפעילו את Docker ונסו שוב".
- [ ] **Step 2–4:** Run (FAIL), implement, run (PASS).
- [ ] **Step 5: Manual check** with `uv run uvicorn chesspgn.web.app:app` and a fake-reader dev flag `CHESSPGN_FAKE_READER=tests/fixtures`:
  - convert Razov;
  - review: flags are highlighted, a pin re-aligns the game, the download filename is correct;
  - RTL: no line mixes directions; the board shows a1 bottom-left.

  **Commit** `feat: Hebrew web UI`.

### Task 12: Docker packaging

**Files:** Create `docker/Dockerfile.app`, `docker/Dockerfile.ollama`, `docker/docker-compose.yml`, `docker/Start Chess PGN.bat`, and `docs/INSTALL.he.md` (the father's install steps, in Hebrew).

**Interfaces:**
- Produces:
  - Image `ghcr.io/<owner>/chesspgn-app` (python:3.12-slim + `tesseract-ocr` + uv sync).
  - Image `ghcr.io/<owner>/chesspgn-ollama` (`FROM ollama/ollama`; at build time, start `ollama serve` in the background and run `ollama pull glm-ocr`, so the model is in the image).
  - Compose: service `ollama` (internal only) and service `app` (port `127.0.0.1:8321:8000`, volume `${USERPROFILE}/Documents/Chess PGN:/data`, `depends_on: ollama`, `restart: unless-stopped`, `cpus: 4`).
  - The `.bat` runs `docker compose -f "%~dp0docker-compose.yml" up -d`, waits for `http://localhost:8321/` (curl retry loop, 120 s), then runs `start http://localhost:8321/`.

- [ ] **Step 1:** Build both images for `linux/amd64` with `docker buildx build --platform linux/amd64`.
- [ ] **Step 2: Smoke test** on the Mac: `docker compose up`, convert all three fixture images through the UI, and confirm the main lines are exact.
- [ ] **Step 3: Speed check:** time each page with `cpus: 4`. Expected: ≤ 120 s per page (Global Constraints). If it's over, record which stage is slow and stop; that's a decision for Ido.
- [ ] **Step 4:** Push the images to GHCR and write `INSTALL.he.md` (install Docker Desktop, turn on "start at login", copy the folder, double-click the shortcut). **Commit** `build: docker packaging`.

### Task 13: Held-out acceptance

**Files:** Create `tests/test_holdout.py` (marked `holdout`, skipped by default) and `docs/acceptance-2026-xx.md`.

- [ ] **Step 1:** Write the test: for each `examples/holdout/*.jpg|png` with a matching `.pgn` (same stem), run `convert` with the real `OllamaReader` and record the main-line prefix match, side-line match, flags, and whether any wrong move is unflagged.
- [ ] **Step 2: Run once:** `uv run pytest -m holdout -s` with Ollama running.
- [ ] **Step 3:** Write the results table to `docs/acceptance-…md`. **Any unflagged wrong move is a failure**, to be reported to Ido before anything is changed. Don't tune on these pages without his say. **Commit** `test: held-out acceptance results`.
