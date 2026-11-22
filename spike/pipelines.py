"""The two candidate pipelines, each with the same legality-repair loop."""

from dataclasses import dataclass, field
from pathlib import Path

import ollama_client
import pgn_check
import prompts
from ollama_client import ChatSettings


@dataclass
class RunLog:
    """Everything a run produced, for the report and for eyeballing."""

    pgn: str = ""
    ocr_text: str = ""
    seconds_by_step: dict[str, float] = field(default_factory=dict)
    repairs: int = 0
    first_problem_each_round: list[str] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return sum(self.seconds_by_step.values())


def ocr_then_llm(
    images: list[Path], *, ocr: ChatSettings, llm: ChatSettings, max_repairs: int
) -> RunLog:
    log = RunLog()
    pages: list[str] = []
    for number, image in enumerate(images, start=1):
        result = ollama_client.chat(ocr, prompts.OCR_PROMPT, [image])
        log.seconds_by_step[f"ocr_page{number}"] = result.seconds
        pages.append(result.text)
    log.ocr_text = "\n\n".join(pages)

    result = ollama_client.chat(llm, prompts.NORMALIZE_FROM_TEXT.replace("@@OCR_TEXT@@", log.ocr_text), [])
    log.seconds_by_step["normalize"] = result.seconds
    log.pgn = pgn_check.clean_model_output(result.text)

    source = prompts.REPAIR_TEXT_SOURCE.format(ocr_text=log.ocr_text)
    _repair_loop(log, llm, images=[], source_suffix=source, max_repairs=max_repairs)
    return log


def direct_vlm(images: list[Path], *, vlm: ChatSettings, max_repairs: int) -> RunLog:
    log = RunLog()
    result = ollama_client.chat(vlm, prompts.DIRECT_FROM_IMAGE, images)
    log.seconds_by_step["image_to_pgn"] = result.seconds
    log.pgn = pgn_check.clean_model_output(result.text)
    _repair_loop(log, vlm, images=images, source_suffix="", max_repairs=max_repairs)
    return log


def _repair_loop(
    log: RunLog, model: ChatSettings, *, images: list[Path], source_suffix: str, max_repairs: int
) -> None:
    for round_number in range(1, max_repairs + 1):
        found = pgn_check.problems(log.pgn)
        if not found:
            return
        log.first_problem_each_round.append(found[0].message.split(" in ")[0])
        prompt = prompts.REPAIR.format(problem=found[0].describe(), pgn=log.pgn) + source_suffix
        result = ollama_client.chat(model, prompt, images)
        log.seconds_by_step[f"repair{round_number}"] = result.seconds
        log.pgn = pgn_check.clean_model_output(result.text)
        log.repairs = round_number
