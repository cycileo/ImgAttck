from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from imgattck.artifacts import create_run_dir, snapshot_config, write_csv, write_json
from imgattck.config import ExperimentConfig, load_config, to_dict
from imgattck.images import initial_image_tensor, make_pil_image, save_rgb_tensor
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
from imgattck.prompting import get_tokenizer, manual_prompt_inputs, move_batch, native_processor_inputs
from imgattck.tokens import check_target_strings, require_single_token_targets, token_report


def check_tokens(config_path: str | Path) -> Path:
    config = load_config(config_path)
    tokenizer = load_tokenizer(config.model)
    checks = check_target_strings(tokenizer, config.target_strings)
    run_dir = create_run_dir(config, "check-tokens")
    snapshot_config(run_dir, config)
    write_json(run_dir / "token_report.json", token_report(checks))
    return run_dir


def optimize_pixels(config_path: str | Path) -> Path:
    config = load_config(config_path)
    spec = spec_from_config(config.image)
    validate_spec(spec)
    bundle = load_model_bundle(config.model)
    target_ids = require_single_token_targets(bundle.tokenizer, config.target_strings)

    run_dir = create_run_dir(config, "pixel")
    snapshot_config(run_dir, config)
    write_json(run_dir / "token_report.json", token_report(check_target_strings(bundle.tokenizer, config.target_strings)))

    base_inputs = move_batch(
        manual_prompt_inputs(
            bundle.tokenizer,
            config.prompt.text,
            spec,
            add_generation_prompt=config.prompt.add_generation_prompt,
            enable_thinking=config.prompt.enable_thinking,
        ),
        bundle.device,
    )
    init_image = initial_image_tensor(
        config.optimization.init, spec, config.optimization.seed, config.optimization.init_image
    ).to(bundle.device)
    parameter = torch.nn.Parameter(rgb_to_image_parameter(init_image))
    optimizer = torch.optim.Adam([parameter], lr=config.optimization.lr)

    rows: list[dict[str, Any]] = []
    progress = tqdm(range(config.optimization.steps), desc="Optimizing pixels")
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        image = image_parameter_to_rgb(parameter)
        processed = differentiable_qwen_preprocess(image, spec)
        batch = dict(base_inputs)
        batch["pixel_values"] = processed.pixel_values
        batch["image_grid_thw"] = processed.image_grid_thw
        logits = next_token_logits(bundle.model, batch)
        loss, metrics = target_probability_loss(logits, target_ids)
        reg_loss = _regularization_loss(image, init_image, config)
        total_loss = loss + reg_loss
        total_loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % config.optimization.log_every == 0 or step + 1 == config.optimization.steps:
            row = {"step": step + 1, **metrics.__dict__, "regularization_loss": float(reg_loss.detach().cpu())}
            rows.append(row)
            progress.set_postfix(target_probability=f"{metrics.target_probability:.4g}", loss=f"{metrics.loss:.4g}")

    final_image = image_parameter_to_rgb(parameter).detach()
    final_path = run_dir / "optimized.png"
    save_rgb_tensor(final_image, final_path)
    write_csv(run_dir / "metrics.csv", rows)
    write_json(run_dir / "metrics.json", rows)
    native = validate_native_image(config, final_path, run_dir=run_dir, bundle=bundle)
    write_json(run_dir / "summary.json", {"final_image": str(final_path), "native_validation": native})
    return run_dir


def validate_native(config_path: str | Path, image_path: str | Path) -> Path:
    config = load_config(config_path)
    bundle = load_model_bundle(config.model)
    run_dir = create_run_dir(config, "native")
    snapshot_config(run_dir, config)
    write_json(run_dir / "token_report.json", token_report(check_target_strings(bundle.tokenizer, config.target_strings)))
    validate_native_image(config, image_path, run_dir=run_dir, bundle=bundle)
    return run_dir


def validate_native_image(
    config: ExperimentConfig,
    image_path: str | Path,
    run_dir: Path,
    bundle: Any,
) -> dict[str, Any]:
    target_ids = require_single_token_targets(bundle.tokenizer, config.target_strings)
    image = Image.open(image_path).convert("RGB")
    spec = spec_from_config(config.image)
    batch = native_processor_inputs(
        bundle.processor,
        config.prompt.text,
        image,
        spec,
        add_generation_prompt=config.prompt.add_generation_prompt,
        enable_thinking=config.prompt.enable_thinking,
    )
    batch = move_batch(batch, bundle.device)
    with torch.no_grad():
        logits = next_token_logits(bundle.model, batch)
        _, metrics = target_probability_loss(logits, target_ids)
    result = {
        "image": str(image_path),
        "metrics": metrics.__dict__,
        "top_tokens": topk_tokens(logits, bundle.tokenizer),
    }
    write_json(run_dir / "native_validation.json", result)
    return result


def optimize_latent(config_path: str | Path) -> Path:
    config = load_config(config_path)
    spec = spec_from_config(config.image)
    validate_spec(spec)
    bundle = load_model_bundle(config.model)
    target_ids = require_single_token_targets(bundle.tokenizer, config.target_strings)

    run_dir = create_run_dir(config, "latent")
    snapshot_config(run_dir, config)
    write_json(run_dir / "token_report.json", token_report(check_target_strings(bundle.tokenizer, config.target_strings)))

    base_inputs = move_batch(
        manual_prompt_inputs(
            bundle.tokenizer,
            config.prompt.text,
            spec,
            add_generation_prompt=config.prompt.add_generation_prompt,
            enable_thinking=config.prompt.enable_thinking,
        ),
        bundle.device,
    )
    init_image = initial_image_tensor(
        config.optimization.init, spec, config.optimization.seed, config.optimization.init_image
    ).to(bundle.device)
    with torch.no_grad():
        processed = differentiable_qwen_preprocess(init_image, spec)
        latent_init = image_features(bundle.model, processed.pixel_values, processed.image_grid_thw).detach()
    latent = torch.nn.Parameter(latent_init.clone())
    optimizer = torch.optim.Adam([latent], lr=config.optimization.lr)

    rows: list[dict[str, Any]] = []
    progress = tqdm(range(config.optimization.steps), desc="Optimizing latent")
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        logits = next_token_logits_from_image_embeds(
            bundle.model,
            input_ids=base_inputs["input_ids"],
            attention_mask=base_inputs["attention_mask"],
            mm_token_type_ids=base_inputs["mm_token_type_ids"],
            image_grid_thw=base_inputs["image_grid_thw"],
            image_embeds=latent,
        )
        loss, metrics = target_probability_loss(logits, target_ids)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % config.optimization.log_every == 0 or step + 1 == config.optimization.steps:
            rows.append({"step": step + 1, **metrics.__dict__})
            progress.set_postfix(target_probability=f"{metrics.target_probability:.4g}", loss=f"{metrics.loss:.4g}")

    torch.save(
        {
            "latent": latent.detach().cpu(),
            "image_grid_thw": base_inputs["image_grid_thw"].detach().cpu(),
            "config": to_dict(config),
        },
        run_dir / "latent.pt",
    )
    write_csv(run_dir / "metrics.csv", rows)
    write_json(run_dir / "metrics.json", rows)
    return run_dir


def invert_latent(config_path: str | Path, latent_path: str | Path) -> Path:
    config = load_config(config_path)
    spec = spec_from_config(config.image)
    validate_spec(spec)
    bundle = load_model_bundle(config.model)

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
    progress = tqdm(range(config.optimization.steps), desc="Inverting latent")
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        image = image_parameter_to_rgb(parameter)
        processed = differentiable_qwen_preprocess(image, spec)
        current_latent = image_features(bundle.model, processed.pixel_values, processed.image_grid_thw)
        feature_loss = torch.nn.functional.mse_loss(current_latent.float(), target_latent.float())
        reg_loss = _regularization_loss(image, init_image, config)
        loss = config.optimization.latent_match_weight * feature_loss + reg_loss
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % config.optimization.log_every == 0 or step + 1 == config.optimization.steps:
            rows.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach().cpu()),
                    "feature_loss": float(feature_loss.detach().cpu()),
                    "regularization_loss": float(reg_loss.detach().cpu()),
                }
            )
            progress.set_postfix(feature_loss=f"{float(feature_loss.detach().cpu()):.4g}")

    final_image = image_parameter_to_rgb(parameter).detach()
    final_path = run_dir / "inverted.png"
    save_rgb_tensor(final_image, final_path)
    write_csv(run_dir / "metrics.csv", rows)
    write_json(run_dir / "metrics.json", rows)
    native = validate_native_image(config, final_path, run_dir=run_dir, bundle=bundle)
    write_json(run_dir / "summary.json", {"final_image": str(final_path), "native_validation": native})
    return run_dir


def _regularization_loss(image: torch.Tensor, init_image: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    loss = image.new_zeros(())
    if config.optimization.tv_weight:
        loss = loss + config.optimization.tv_weight * total_variation(image)
    if config.optimization.l2_weight:
        loss = loss + config.optimization.l2_weight * torch.nn.functional.mse_loss(image, init_image)
    return loss
