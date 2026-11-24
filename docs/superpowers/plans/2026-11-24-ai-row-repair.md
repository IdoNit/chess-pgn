# AI row repair — plan

Approved design (chat, 2026-11-24): after the normal conversion, rows whose moves the board could not read get a
second reading from a small vision model (qwen3-vl 8b), one cropped row at a time. A reading is used only if it is
close to what the OCR saw; the board re-reads the game with it; moves from it are flagged "read by AI".
On by default; switched off automatically below ~16 GB memory; model shipped inside the installer.

Measured basis: Capablanca game 3 goes from 47/72 to 69/72 moves right, no new silent errors elsewhere
(bench --vlm repair, saved answers). Speed on CPU ~15-45 s per question; only flagged rows are asked.

## Tasks (each test-first)
1. `tokens`: a move written with a trailing "†" is an AI reading -> `Token.ai`; the decoder flags it `ai_read`
   (persistent: the stored text carries the marker, so re-decoding after a fix keeps the flag).
2. `repair.py`: find a line's box on the page (tesseract), crop it, ask the row reader, check closeness and
   readability, rewrite the line as "N. white† black†". `RowReader` protocol + `OllamaRowReader`.
3. `pipeline.convert(..., row_reader=None, progress=None)`: second pass only when the first leaves trouble flags
   (unreadable / missing / skipped) on the main line; asks only about those move numbers.
4. Web: settings (on by default; off below 16 GB unless switched on), GET/PUT /api/settings, progress line during
   convert (GET /api/progress), Hebrew strings, "read by AI" reason.
5. Docker: bake qwen3-vl:8b-instruct into the ollama image; app knows the model name.
6. Bench: run the real repair path with a reader that replays the saved answers; Capablanca: game 3 rescued, no
   new silent errors. Full suite green.
