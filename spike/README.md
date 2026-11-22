# Spike: which pipeline reads chess pages best? (throwaway)

Compares, on the pages in `../examples`:

- **A**: `glm-ocr` reads the page text, then `qwen3-vl:8b-instruct` turns the text into PGN
- **B**: `qwen3-vl:8b-instruct` looks at the image and writes the PGN directly

Both get the same rules and the same repair loop: python-chess replays every move, and an illegal move is
sent back to the model with the position and the list of legal moves (up to 2 times).

## Run it (on the Mac)

1. Install / update Ollama: https://ollama.com/download (GLM-OCR needs a recent version). Keep the app running.
2. Download the models (about 2.2 GB + 6 GB):
   ```
   ollama pull glm-ocr
   ollama pull qwen3-vl:8b-instruct
   ```
3. From the project folder:
   ```
   uv run spike/run_spike.py
   ```
   (no uv? `python3 -m pip install chess` then `python3 spike/run_spike.py`)
4. Then the CPU-only run (slower — this is the number that matters for a PC without a graphics card):
   ```
   uv run spike/run_spike.py --cpu
   ```

Results land in `spike/results/<timestamp>/`: a `report.md` table, plus each model's PGN, the OCR text,
and timings per step.

Useful options: `--pipelines A` (only one), `--llm qwen3-vl:4b-instruct --vlm qwen3-vl:4b-instruct` (smaller model),
`--no-think-flag` (if Ollama complains that a model doesn't support thinking).

Use the `-instruct` tags: the plain `qwen3-vl:8b` tag is a thinking model that spends its whole output budget thinking and returns an empty answer.
