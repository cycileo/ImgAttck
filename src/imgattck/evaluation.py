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
from imgattck.modeling import ModelBundle, load_model_bundle, next_token_logits
from imgattck.preprocess import spec_from_config, validate_spec
from imgattck.prompting import move_batch, native_processor_inputs, text_processor_inputs
from imgattck.tokens import check_target_strings, token_report


def evaluate_image(config_path: str | Path) -> Path:
    config = load_evaluation_config(config_path)
    _validate_success_config(config.success)
    spec = spec_from_config(config.image_spec)
    validate_spec(spec)

    image_path = resolve_evaluation_image_path(config)
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
                    "answer": result["with_image"]["answer"],
                    "answer_without_image": result["without_image"]["answer"],
                    "answer_with_image": result["with_image"]["answer"],
                    "success": result["with_image"]["success"],
                    "success_without_image": result["without_image"]["success"],
                    "success_with_image": result["with_image"]["success"],
                    "success_strings": "|".join(result["success_strings"]),
                    "target_strings": "|".join(result["target_strings"]),
                    "target_probability_without_image": result["target_probability_without_image"],
                    "target_probability_with_image": result["target_probability_with_image"],
                    "target_probability_delta": result["target_probability_delta"],
                    "target_log_probability_without_image": result["target_log_probability_without_image"],
                    "target_log_probability_with_image": result["target_log_probability_with_image"],
                    "target_log_probability_delta": result["target_log_probability_delta"],
                    "token_probability_deltas": "|".join(
                        f"{item['text']}:{item['probability_delta']:.8g}" for item in result["token_comparison"]
                    ),
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


def resolve_evaluation_image_path(config: EvaluationConfig) -> Path:
    if config.image:
        image_path = Path(config.image).expanduser()
        if not image_path.exists():
            raise FileNotFoundError(f"Evaluation image does not exist: {image_path}")
        return image_path

    run_root = Path(config.output.root).expanduser()
    candidates = sorted(
        path / "optimized.png"
        for path in run_root.glob("pixel-*")
        if path.is_dir() and (path / "optimized.png").exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No evaluation image was specified and no optimized image was found under {run_root}/pixel-*/optimized.png."
        )
    return candidates[-1]


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
    without_image_batch = text_processor_inputs(
        bundle.processor,
        question.text,
        add_generation_prompt=config.prompt.add_generation_prompt,
        enable_thinking=config.prompt.enable_thinking,
    )
    without_image = _run_inference(bundle, without_image_batch, config.generation)

    with_image_batch = native_processor_inputs(
        bundle.processor,
        question.text,
        image,
        spec,
        add_generation_prompt=config.prompt.add_generation_prompt,
        enable_thinking=config.prompt.enable_thinking,
    )
    with_image = _run_inference(bundle, with_image_batch, config.generation)

    success_strings = question.success_strings or config.success.strings
    target_strings = _target_strings_for_question(config, question, success_strings)
    target_metrics = compare_target_tokens(
        bundle.tokenizer,
        without_image["logits"],
        with_image["logits"],
        target_strings,
    )

    without_answer = str(without_image["answer"])
    with_answer = str(with_image["answer"])
    without_success = answer_matches(without_answer, success_strings, config.success)
    with_success = answer_matches(with_answer, success_strings, config.success)
    return {
        "question_index": question_index,
        "question": question.text,
        "success_strings": success_strings,
        "target_strings": target_strings,
        "without_image": {
            "answer": without_answer,
            "success": without_success,
            "target_probability": target_metrics["target_probability_without_image"],
            "target_log_probability": target_metrics["target_log_probability_without_image"],
        },
        "with_image": {
            "answer": with_answer,
            "success": with_success,
            "target_probability": target_metrics["target_probability_with_image"],
            "target_log_probability": target_metrics["target_log_probability_with_image"],
        },
        "answer": with_answer,
        "success": with_success,
        **target_metrics,
    }


def _run_inference(bundle: ModelBundle, batch: dict[str, torch.Tensor], config: GenerationConfig) -> dict[str, Any]:
    input_length = int(batch["input_ids"].shape[-1])
    batch = move_batch(batch, bundle.device)
    with torch.no_grad():
        logits = next_token_logits(bundle.model, batch)
        output_ids = bundle.model.generate(**batch, **_generation_kwargs(config))

    generated_ids = output_ids[:, input_length:] if output_ids.shape[-1] > input_length else output_ids
    answer = bundle.tokenizer.decode(
        generated_ids[0].detach().cpu(),
        skip_special_tokens=config.skip_special_tokens,
    )
    return {
        "answer": answer,
        "logits": logits.detach().float().cpu(),
    }


def compare_target_tokens(
    tokenizer: object,
    logits_without_image: torch.Tensor,
    logits_with_image: torch.Tensor,
    target_strings: list[str],
) -> dict[str, Any]:
    checks = check_target_strings(tokenizer, target_strings)
    failures = [check for check in checks if not check.is_single_token]
    if failures:
        details = "; ".join(f"{item.text!r} -> {item.token_ids}" for item in failures)
        raise ValueError(f"Evaluation target_strings must encode to exactly one token: {details}")

    target_token_ids = [check.token_ids[0] for check in checks]
    if not target_token_ids:
        raise ValueError("Evaluation target_strings must contain at least one target.")

    without_logits = logits_without_image.reshape(-1).float()
    with_logits = logits_with_image.reshape(-1).float()
    without_log_probs = torch.log_softmax(without_logits, dim=-1)
    with_log_probs = torch.log_softmax(with_logits, dim=-1)

    unique_target_ids = torch.tensor(sorted(set(target_token_ids)), dtype=torch.long)
    without_target_log_probability = torch.logsumexp(without_log_probs[unique_target_ids], dim=0)
    with_target_log_probability = torch.logsumexp(with_log_probs[unique_target_ids], dim=0)
    without_target_probability = float(without_target_log_probability.exp().item())
    with_target_probability = float(with_target_log_probability.exp().item())

    rows: list[dict[str, Any]] = []
    for check in checks:
        token_id = check.token_ids[0]
        logit_without = float(without_logits[token_id].item())
        logit_with = float(with_logits[token_id].item())
        probability_without = float(without_log_probs[token_id].exp().item())
        probability_with = float(with_log_probs[token_id].exp().item())
        probability_ratio = None if probability_without == 0.0 else probability_with / probability_without
        rows.append(
            {
                "text": check.text,
                "token_id": token_id,
                "decoded": check.decoded,
                "logit_without_image": logit_without,
                "logit_with_image": logit_with,
                "logit_delta": logit_with - logit_without,
                "probability_without_image": probability_without,
                "probability_with_image": probability_with,
                "probability_delta": probability_with - probability_without,
                "probability_ratio": probability_ratio,
            }
        )

    return {
        "token_report": token_report(checks),
        "token_comparison": rows,
        "target_probability_without_image": without_target_probability,
        "target_probability_with_image": with_target_probability,
        "target_probability_delta": with_target_probability - without_target_probability,
        "target_log_probability_without_image": float(without_target_log_probability.item()),
        "target_log_probability_with_image": float(with_target_log_probability.item()),
        "target_log_probability_delta": float(
            (with_target_log_probability - without_target_log_probability).item()
        ),
    }


def _target_strings_for_question(
    config: EvaluationConfig,
    question: EvaluationQuestion,
    success_strings: list[str],
) -> list[str]:
    if question.target_strings:
        return question.target_strings
    if config.target_strings:
        return config.target_strings
    if question.success_strings:
        return question.success_strings
    return success_strings


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
