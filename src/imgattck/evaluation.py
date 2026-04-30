from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from imgattck.artifacts import create_run_dir, snapshot_config, write_csv, write_json
from imgattck.config import (
    EvaluationConfig,
    EvaluationQuestion,
    GenerationConfig,
    SuccessConfig,
    load_evaluation_config,
)
from imgattck.modeling import ModelBundle, load_model_bundle
from imgattck.preprocess import spec_from_config, validate_spec
from imgattck.prompting import move_batch, native_processor_inputs


def evaluate_image(config_path: str | Path) -> Path:
    config = load_evaluation_config(config_path)
    _validate_success_config(config.success)
    spec = spec_from_config(config.image_spec)
    validate_spec(spec)

    image_path = Path(config.image).expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"Evaluation image does not exist: {image_path}")
    image = Image.open(image_path).convert("RGB")

    run_dir = create_run_dir(config, "eval")
    snapshot_config(run_dir, config)

    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for model_index, model_config in enumerate(tqdm(config.models, desc="Evaluating models")):
        bundle = load_model_bundle(model_config)
        model_results: list[dict[str, Any]] = []
        try:
            for question_index, question in enumerate(config.questions):
                result = _evaluate_question(bundle, image, spec, config, question, question_index)
                row = {
                    "model_index": model_index,
                    "model": model_config.name,
                    "question_index": question_index,
                    "question": question.text,
                    "answer": result["answer"],
                    "success": result["success"],
                    "success_strings": "|".join(result["success_strings"]),
                }
                rows.append(row)
                model_results.append(result)
        finally:
            del bundle
            _release_model_memory()

        results.append(
            {
                "model_index": model_index,
                "model": model_config.name,
                "answers": model_results,
            }
        )

    summary = summarize_results(rows, total_models=len(config.models), total_questions=len(config.questions))
    write_json(run_dir / "results.json", {"image": str(image_path), "models": results})
    write_csv(run_dir / "results.csv", rows)
    write_json(run_dir / "summary.json", summary)
    return run_dir


def summarize_results(
    rows: list[dict[str, Any]],
    total_models: int | None = None,
    total_questions: int | None = None,
) -> dict[str, Any]:
    total_trials = len(rows)
    total_successes = sum(1 for row in rows if row["success"])
    summary = {
        "total_models": total_models,
        "total_questions": total_questions,
        "total_trials": total_trials,
        "successes": total_successes,
        "success_rate": _success_rate(total_successes, total_trials),
        "by_model": [],
        "by_question": [],
    }

    model_keys = [(row["model_index"], row["model"]) for row in rows]
    for model_index, model in _unique_in_order(model_keys):
        model_rows = [row for row in rows if row["model_index"] == model_index]
        successes = sum(1 for row in model_rows if row["success"])
        summary["by_model"].append(
            {
                "model_index": model_index,
                "model": model,
                "successes": successes,
                "total": len(model_rows),
                "success_rate": _success_rate(successes, len(model_rows)),
            }
        )

    question_keys = [(row["question_index"], row["question"]) for row in rows]
    for question_index, question in _unique_in_order(question_keys):
        question_rows = [row for row in rows if row["question_index"] == question_index]
        successes = sum(1 for row in question_rows if row["success"])
        summary["by_question"].append(
            {
                "question_index": question_index,
                "question": question,
                "successes": successes,
                "total": len(question_rows),
                "success_rate": _success_rate(successes, len(question_rows)),
            }
        )

    return summary


def answer_matches(answer: str, success_strings: list[str], config: SuccessConfig) -> bool:
    _validate_success_config(config)
    candidate = answer.strip() if config.strip else answer
    if not config.case_sensitive:
        candidate = candidate.casefold()

    for success_string in success_strings:
        needle = success_string.strip() if config.strip else success_string
        if not needle:
            continue
        if not config.case_sensitive:
            needle = needle.casefold()
        if config.mode == "contains" and needle in candidate:
            return True
        if config.mode in {"prefix", "startswith"} and candidate.startswith(needle):
            return True
        if config.mode == "exact" and candidate == needle:
            return True
    return False


def _evaluate_question(
    bundle: ModelBundle,
    image: Image.Image,
    spec: Any,
    config: EvaluationConfig,
    question: EvaluationQuestion,
    question_index: int,
) -> dict[str, Any]:
    batch = native_processor_inputs(
        bundle.processor,
        question.text,
        image,
        spec,
        add_generation_prompt=config.prompt.add_generation_prompt,
        enable_thinking=config.prompt.enable_thinking,
    )
    input_length = int(batch["input_ids"].shape[-1])
    batch = move_batch(batch, bundle.device)
    with torch.no_grad():
        output_ids = bundle.model.generate(**batch, **_generation_kwargs(config.generation))

    generated_ids = output_ids[:, input_length:] if output_ids.shape[-1] > input_length else output_ids
    answer = bundle.tokenizer.decode(
        generated_ids[0].detach().cpu(),
        skip_special_tokens=config.generation.skip_special_tokens,
    )
    success_strings = question.success_strings or config.success.strings
    success = answer_matches(answer, success_strings, config.success)
    return {
        "question_index": question_index,
        "question": question.text,
        "answer": answer,
        "success": success,
        "success_strings": success_strings,
    }


def _generation_kwargs(config: GenerationConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
    }
    optional_values = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "num_beams": config.num_beams,
    }
    for key, value in optional_values.items():
        if value is not None:
            kwargs[key] = value
    return kwargs


def _validate_success_config(config: SuccessConfig) -> None:
    valid_modes = {"contains", "exact", "prefix", "startswith"}
    if config.mode not in valid_modes:
        modes = ", ".join(sorted(valid_modes))
        raise ValueError(f"Unknown success.mode {config.mode!r}; expected one of: {modes}.")


def _success_rate(successes: int, total: int) -> float:
    if total == 0:
        return 0.0
    return successes / total


def _unique_in_order(values: list[tuple[int, str]]) -> list[tuple[int, str]]:
    seen: set[tuple[int, str]] = set()
    unique: list[tuple[int, str]] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _release_model_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.empty_cache()
