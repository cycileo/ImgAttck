from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from imgattck.artifacts import create_run_dir, snapshot_config, write_csv, write_json
from imgattck.config import (
    ExperimentConfig,
    PromptConfig,
    experiment_model_configs,
    experiment_prompt_configs,
    load_config,
    to_dict,
)
from imgattck.images import initial_image_tensor, save_rgb_tensor
from imgattck.losses import target_probability_loss, topk_tokens
from imgattck.modeling import (
    image_features,
    load_model_bundle,
    load_tokenizer,
    next_token_logits,
    next_token_logits_from_image_embeds,
)
from imgattck.preprocess import (
    differentiable_qwen_preprocess,
    image_parameter_to_rgb,
    rgb_to_image_parameter,
    spec_from_config,
    total_variation,
    validate_spec,
)
from imgattck.prompting import manual_prompt_inputs, move_batch, native_processor_inputs
from imgattck.tokens import check_target_strings, require_single_token_targets, token_report


@dataclass
class OptimizationContext:
    label: str
    bundle: Any
    target_ids: list[int]
    base_inputs: dict[str, torch.Tensor]
    prompt_index: int
    prompt: PromptConfig


def check_tokens(config_path: str | Path) -> Path:
    config = load_config(config_path)
    run_dir = create_run_dir(config, "check-tokens")
    snapshot_config(run_dir, config)
    write_json(run_dir / "token_report.json", _token_reports(config))
    return run_dir


def optimize_pixels(config_path: str | Path) -> Path:
    config = load_config(config_path)
    spec = spec_from_config(config.image)
    validate_spec(spec)
    contexts = _optimization_contexts(config, spec)

    run_dir = create_run_dir(config, "pixel")
    snapshot_config(run_dir, config)
    write_json(run_dir / "token_report.json", _token_reports_from_contexts(config, contexts))

    init_image = initial_image_tensor(
        config.optimization.init, spec, config.optimization.seed, config.optimization.init_image
    ).to(contexts[0].bundle.device)
    parameter = torch.nn.Parameter(rgb_to_image_parameter(init_image))
    optimizer = torch.optim.Adam([parameter], lr=config.optimization.lr)

    rows: list[dict[str, Any]] = []
    stopped_early = False
    stop_reason = None
    progress = tqdm(range(config.optimization.steps), desc="Optimizing pixels")
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        model_rows: list[dict[str, Any]] = []
        data_loss_value = 0.0
        data_target_probability = 0.0
        for context in contexts:
            image = image_parameter_to_rgb(parameter).to(context.bundle.device)
            processed = differentiable_qwen_preprocess(image, spec)
            batch = dict(context.base_inputs)
            batch["pixel_values"] = processed.pixel_values
            batch["image_grid_thw"] = processed.image_grid_thw
            logits = next_token_logits(context.bundle.model, batch)
            loss, metrics = target_probability_loss(logits, context.target_ids)
            (loss / len(contexts)).backward()
            data_loss_value += metrics.loss / len(contexts)
            data_target_probability += metrics.target_probability / len(contexts)
            model_rows.append(
                {
                    "model": context.label,
                    "prompt_index": context.prompt_index,
                    "prompt": context.prompt.text,
                    **metrics.__dict__,
                }
            )

        image = image_parameter_to_rgb(parameter)
        reg_loss = _regularization_loss(image, init_image, config)
        if reg_loss.requires_grad:
            reg_loss.backward()
        total_loss_value = data_loss_value + float(reg_loss.detach().cpu())

        should_log = step == 0 or (step + 1) % config.optimization.log_every == 0 or step + 1 == config.optimization.steps
        should_stop = total_loss_value <= config.optimization.early_stop_loss
        if should_log or should_stop:
            row = {
                "step": step + 1,
                "loss": total_loss_value,
                "target_loss": data_loss_value,
                "target_probability": data_target_probability,
                "regularization_loss": float(reg_loss.detach().cpu()),
                "stopped_early": should_stop,
            }
            rows.append(row)
            write_json(run_dir / f"step_{step + 1:05d}_models.json", model_rows)
            progress.set_postfix(target_probability=f"{data_target_probability:.4g}", loss=f"{total_loss_value:.4g}")
        if should_stop:
            stopped_early = True
            stop_reason = f"loss <= early_stop_loss ({config.optimization.early_stop_loss})"
            break
        optimizer.step()

    final_image = image_parameter_to_rgb(parameter).detach()
    final_path = run_dir / "optimized.png"
    save_rgb_tensor(final_image, final_path)
    write_csv(run_dir / "metrics.csv", rows)
    write_json(run_dir / "metrics.json", rows)
    native = validate_native_image(config, final_path, run_dir=run_dir, bundles=_unique_bundles(contexts))
    write_json(
        run_dir / "summary.json",
        {
            "final_image": str(final_path),
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "native_validation": native,
        },
    )
    return run_dir


def validate_native(config_path: str | Path, image_path: str | Path) -> Path:
    config = load_config(config_path)
    bundles = [load_model_bundle(model_config) for model_config in experiment_model_configs(config)]
    run_dir = create_run_dir(config, "native")
    snapshot_config(run_dir, config)
    write_json(run_dir / "token_report.json", _token_reports_from_bundles(config, bundles))
    validate_native_image(config, image_path, run_dir=run_dir, bundles=bundles)
    return run_dir


def validate_native_image(
    config: ExperimentConfig,
    image_path: str | Path,
    run_dir: Path,
    bundles: list[Any],
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    spec = spec_from_config(config.image)
    prompts = experiment_prompt_configs(config)
    model_results = []
    for bundle in bundles:
        target_ids = require_single_token_targets(bundle.tokenizer, config.target_strings)
        prompt_results = []
        for prompt_index, prompt in enumerate(prompts):
            batch = native_processor_inputs(
                bundle.processor,
                prompt.text,
                image,
                spec,
                add_generation_prompt=prompt.add_generation_prompt,
                enable_thinking=prompt.enable_thinking,
            )
            batch = move_batch(batch, bundle.device)
            with torch.no_grad():
                logits = next_token_logits(bundle.model, batch)
                _, metrics = target_probability_loss(logits, target_ids)
            prompt_results.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": prompt.text,
                    "metrics": metrics.__dict__,
                    "top_tokens": topk_tokens(logits, bundle.tokenizer),
                }
            )
        model_result = {
            "model": bundle.model.config.name_or_path,
            "prompts": prompt_results,
        }
        if len(prompt_results) == 1:
            model_result.update(prompt_results[0])
        model_results.append(model_result)
    result = {"image": str(image_path), "models": model_results}
    write_json(run_dir / "native_validation.json", result)
    return result


def optimize_latent(config_path: str | Path) -> Path:
    config = load_config(config_path)
    _require_single_experiment_model(config, "optimize-latent")
    spec = spec_from_config(config.image)
    validate_spec(spec)
    contexts = _optimization_contexts(config, spec)
    bundle = contexts[0].bundle

    run_dir = create_run_dir(config, "latent")
    snapshot_config(run_dir, config)
    write_json(run_dir / "token_report.json", token_report(check_target_strings(bundle.tokenizer, config.target_strings)))
    init_image = initial_image_tensor(
        config.optimization.init, spec, config.optimization.seed, config.optimization.init_image
    ).to(bundle.device)
    with torch.no_grad():
        processed = differentiable_qwen_preprocess(init_image, spec)
        latent_init = image_features(bundle.model, processed.pixel_values, processed.image_grid_thw).detach()
    latent = torch.nn.Parameter(latent_init.clone())
    optimizer = torch.optim.Adam([latent], lr=config.optimization.lr)

    rows: list[dict[str, Any]] = []
    stopped_early = False
    stop_reason = None
    progress = tqdm(range(config.optimization.steps), desc="Optimizing latent")
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        loss = latent.new_zeros(())
        model_rows: list[dict[str, Any]] = []
        data_loss_value = 0.0
        data_target_probability = 0.0
        for context in contexts:
            logits = next_token_logits_from_image_embeds(
                context.bundle.model,
                input_ids=context.base_inputs["input_ids"],
                attention_mask=context.base_inputs["attention_mask"],
                mm_token_type_ids=context.base_inputs["mm_token_type_ids"],
                image_grid_thw=context.base_inputs["image_grid_thw"],
                image_embeds=latent,
            )
            context_loss, metrics = target_probability_loss(logits, context.target_ids)
            loss = loss + context_loss / len(contexts)
            data_loss_value += metrics.loss / len(contexts)
            data_target_probability += metrics.target_probability / len(contexts)
            model_rows.append(
                {
                    "model": context.label,
                    "prompt_index": context.prompt_index,
                    "prompt": context.prompt.text,
                    **metrics.__dict__,
                }
            )

        should_stop = data_loss_value <= config.optimization.early_stop_loss
        if not should_stop:
            loss.backward()
            optimizer.step()
        should_log = step == 0 or (step + 1) % config.optimization.log_every == 0 or step + 1 == config.optimization.steps
        if should_log or should_stop:
            rows.append(
                {
                    "step": step + 1,
                    "loss": data_loss_value,
                    "target_probability": data_target_probability,
                    "stopped_early": should_stop,
                }
            )
            write_json(run_dir / f"step_{step + 1:05d}_prompts.json", model_rows)
            progress.set_postfix(target_probability=f"{data_target_probability:.4g}", loss=f"{data_loss_value:.4g}")
        if should_stop:
            stopped_early = True
            stop_reason = f"loss <= early_stop_loss ({config.optimization.early_stop_loss})"
            break

    torch.save(
        {
            "latent": latent.detach().cpu(),
            "image_grid_thw": contexts[0].base_inputs["image_grid_thw"].detach().cpu(),
            "config": to_dict(config),
        },
        run_dir / "latent.pt",
    )
    write_csv(run_dir / "metrics.csv", rows)
    write_json(run_dir / "metrics.json", rows)
    write_json(run_dir / "summary.json", {"stopped_early": stopped_early, "stop_reason": stop_reason})
    return run_dir


def invert_latent(config_path: str | Path, latent_path: str | Path) -> Path:
    config = load_config(config_path)
    _require_single_experiment_model(config, "invert-latent")
    spec = spec_from_config(config.image)
    validate_spec(spec)
    bundle = load_model_bundle(experiment_model_configs(config)[0])

    checkpoint = torch.load(latent_path, map_location="cpu")
    target_latent = checkpoint["latent"].to(bundle.device)
    run_dir = create_run_dir(config, "invert")
    snapshot_config(run_dir, config)
    write_json(run_dir / "token_report.json", token_report(check_target_strings(bundle.tokenizer, config.target_strings)))

    init_image = initial_image_tensor(
        config.optimization.init, spec, config.optimization.seed, config.optimization.init_image
    ).to(bundle.device)
    parameter = torch.nn.Parameter(rgb_to_image_parameter(init_image))
    optimizer = torch.optim.Adam([parameter], lr=config.optimization.lr)

    rows: list[dict[str, Any]] = []
    stopped_early = False
    stop_reason = None
    progress = tqdm(range(config.optimization.steps), desc="Inverting latent")
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        image = image_parameter_to_rgb(parameter)
        processed = differentiable_qwen_preprocess(image, spec)
        current_latent = image_features(bundle.model, processed.pixel_values, processed.image_grid_thw)
        feature_loss = torch.nn.functional.mse_loss(current_latent.float(), target_latent.float())
        reg_loss = _regularization_loss(image, init_image, config)
        loss = config.optimization.latent_match_weight * feature_loss + reg_loss
        loss_value = float(loss.detach().cpu())
        should_stop = loss_value <= config.optimization.early_stop_loss
        if not should_stop:
            loss.backward()
            optimizer.step()
        should_log = step == 0 or (step + 1) % config.optimization.log_every == 0 or step + 1 == config.optimization.steps
        if should_log or should_stop:
            rows.append(
                {
                    "step": step + 1,
                    "loss": loss_value,
                    "feature_loss": float(feature_loss.detach().cpu()),
                    "regularization_loss": float(reg_loss.detach().cpu()),
                    "stopped_early": should_stop,
                }
            )
            progress.set_postfix(feature_loss=f"{float(feature_loss.detach().cpu()):.4g}")
        if should_stop:
            stopped_early = True
            stop_reason = f"loss <= early_stop_loss ({config.optimization.early_stop_loss})"
            break

    final_image = image_parameter_to_rgb(parameter).detach()
    final_path = run_dir / "inverted.png"
    save_rgb_tensor(final_image, final_path)
    write_csv(run_dir / "metrics.csv", rows)
    write_json(run_dir / "metrics.json", rows)
    native = validate_native_image(config, final_path, run_dir=run_dir, bundles=[bundle])
    write_json(
        run_dir / "summary.json",
        {
            "final_image": str(final_path),
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "native_validation": native,
        },
    )
    return run_dir


def _regularization_loss(image: torch.Tensor, init_image: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    loss = image.new_zeros(())
    if config.optimization.tv_weight:
        loss = loss + config.optimization.tv_weight * total_variation(image)
    if config.optimization.l2_weight:
        loss = loss + config.optimization.l2_weight * torch.nn.functional.mse_loss(image, init_image)
    return loss


def _optimization_contexts(config: ExperimentConfig, spec: Any) -> list[OptimizationContext]:
    contexts = []
    prompts = experiment_prompt_configs(config)
    for index, model_config in enumerate(experiment_model_configs(config)):
        bundle = load_model_bundle(model_config)
        target_ids = require_single_token_targets(bundle.tokenizer, config.target_strings)
        for prompt_index, prompt in enumerate(prompts):
            base_inputs = move_batch(
                manual_prompt_inputs(
                    bundle.tokenizer,
                    prompt.text,
                    spec,
                    add_generation_prompt=prompt.add_generation_prompt,
                    enable_thinking=prompt.enable_thinking,
                ),
                bundle.device,
            )
            contexts.append(
                OptimizationContext(
                    label=_model_label(model_config.name, index),
                    bundle=bundle,
                    target_ids=target_ids,
                    base_inputs=base_inputs,
                    prompt_index=prompt_index,
                    prompt=prompt,
                )
            )
    return contexts


def _token_reports(config: ExperimentConfig) -> dict[str, Any]:
    reports = []
    for index, model_config in enumerate(experiment_model_configs(config)):
        tokenizer = load_tokenizer(model_config)
        reports.append(
            {
                "model": _model_label(model_config.name, index),
                "targets": token_report(check_target_strings(tokenizer, config.target_strings)),
            }
        )
    return {"models": reports}


def _token_reports_from_contexts(config: ExperimentConfig, contexts: list[OptimizationContext]) -> dict[str, Any]:
    reports = []
    seen: set[int] = set()
    for context in contexts:
        identity = id(context.bundle)
        if identity in seen:
            continue
        seen.add(identity)
        reports.append(
            {
                "model": context.label,
                "targets": token_report(check_target_strings(context.bundle.tokenizer, config.target_strings)),
            }
        )
    return {
        "models": reports,
        "prompts": [
            {
                "prompt_index": index,
                "prompt": prompt.text,
                "add_generation_prompt": prompt.add_generation_prompt,
                "enable_thinking": prompt.enable_thinking,
            }
            for index, prompt in enumerate(experiment_prompt_configs(config))
        ],
    }


def _token_reports_from_bundles(config: ExperimentConfig, bundles: list[Any]) -> dict[str, Any]:
    return {
        "models": [
            {
                "model": bundle.model.config.name_or_path,
                "targets": token_report(check_target_strings(bundle.tokenizer, config.target_strings)),
            }
            for bundle in bundles
        ]
    }


def _require_single_experiment_model(config: ExperimentConfig, command: str) -> None:
    models = experiment_model_configs(config)
    if len(models) != 1:
        raise ValueError(
            f"{command} currently supports exactly one model because visual latent shapes are model-specific; "
            f"got {len(models)} models."
        )


def _model_label(model_name: str, index: int) -> str:
    return model_name if index == 0 else f"{model_name}#{index + 1}"


def _unique_bundles(contexts: list[OptimizationContext]) -> list[Any]:
    bundles = []
    seen: set[int] = set()
    for context in contexts:
        identity = id(context.bundle)
        if identity in seen:
            continue
        seen.add(identity)
        bundles.append(context.bundle)
    return bundles
