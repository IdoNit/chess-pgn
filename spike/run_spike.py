# /// script
# requires-python = ">=3.10"
# dependencies = ["chess>=1.10"]
# ///
"""THROWAWAY spike: compare 'OCR model -> LLM' with 'vision model straight to PGN'.

Usage (from the project folder, with the Ollama app running):
    uv run spike/run_spike.py            # GPU (fast) on your Mac
    uv run spike/run_spike.py --cpu      # CPU only, closer to a PC without a graphics card
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pgn_check
import pipelines
from ollama_client import ChatSettings
from pipelines import RunLog

SPIKE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SPIKE_DIR.parent / "examples"


def main() -> int:
    args = _parse_args()
    cases = json.loads((SPIKE_DIR / "cases.json").read_text(encoding="utf-8"))
    out_dir = SPIKE_DIR / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)

    think = None if args.no_think_flag else False
    ocr = ChatSettings(model=args.ocr_model, cpu_only=args.cpu, think=None)
    llm = ChatSettings(model=args.llm, cpu_only=args.cpu, think=think)
    vlm = ChatSettings(model=args.vlm, cpu_only=args.cpu, think=think)

    rows: list[str] = []
    for case in cases:
        images = [EXAMPLES_DIR / name for name in case["images"]]
        expected = (EXAMPLES_DIR / case["expected"]).read_text(encoding="utf-8")
        for pipeline in args.pipelines:
            label = f"A: {args.ocr_model} -> {args.llm}" if pipeline == "A" else f"B: {args.vlm}"
            print(f"[{case['name']}] pipeline {label} ...", flush=True)
            if pipeline == "A":
                log = pipelines.ocr_then_llm(images, ocr=ocr, llm=llm, max_repairs=args.max_repairs)
            else:
                log = pipelines.direct_vlm(images, vlm=vlm, max_repairs=args.max_repairs)
            _save(out_dir, case["name"], pipeline, log)
            rows.append(_row(case["name"], label, log, pgn_check.score(log.pgn, expected)))
            print(f"    done in {log.total_seconds:.0f}s", flush=True)

    report = _report(args, rows)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Outputs saved in {out_dir}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipelines", nargs="+", choices=["A", "B"], default=["A", "B"])
    parser.add_argument("--ocr-model", default="glm-ocr")
    parser.add_argument("--llm", default="qwen3-vl:8b-instruct", help="text normalizer for pipeline A")
    parser.add_argument("--vlm", default="qwen3-vl:8b-instruct", help="image-to-PGN model for pipeline B")
    parser.add_argument("--cpu", action="store_true", help="force CPU (no GPU) to mimic a plain PC")
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument(
        "--no-think-flag",
        action="store_true",
        help="don't send think=false (use if Ollama says the model doesn't support thinking)",
    )
    return parser.parse_args()


def _save(out_dir: Path, case_name: str, pipeline: str, log: RunLog) -> None:
    stem = f"{pipeline}__{case_name.split(' (')[0]}"
    (out_dir / f"{stem}.pgn").write_text(log.pgn, encoding="utf-8")
    if log.ocr_text:
        (out_dir / f"{stem}.ocr.txt").write_text(log.ocr_text, encoding="utf-8")
    details = {
        "seconds_by_step": log.seconds_by_step,
        "repairs": log.repairs,
        "problems_sent_for_repair": log.first_problem_each_round,
        "problems_left": [p.message.split(" in ")[0] for p in pgn_check.problems(log.pgn)],
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(details, indent=2), encoding="utf-8")


def _row(case_name: str, label: str, log: RunLog, s: pgn_check.Score) -> str:
    main = f"{s.main_prefix_ok}/{s.expected_main_moves}" + (" ✅" if s.main_exact else "")
    variations = f"{s.variation_hits}/{s.expected_variation_moves} (made {s.variation_moves})"
    return (
        f"| {case_name} | {label} | {main} | {variations} | {s.problems} | "
        f"{'✅' if s.result_ok else '❌'} | {log.repairs} | {log.total_seconds:.0f}s |"
    )


def _report(args: argparse.Namespace, rows: list[str]) -> str:
    header = (
        f"# Spike results ({'CPU only' if args.cpu else 'GPU'})\n\n"
        "Main line = correct half-moves before the first difference / expected half-moves.\n"
        "Variations = side-line moves matching the expected PGN (the expected files differ in how "
        "ChatGPT handled side lines, so treat this as a rough guide).\n"
        "Illegal = moves python-chess could not play in the final PGN.\n\n"
        "| Game | Pipeline | Main line | Variations | Illegal | Result | Repairs | Time |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    return header + "\n".join(rows) + "\n"


if __name__ == "__main__":
    sys.exit(main())
