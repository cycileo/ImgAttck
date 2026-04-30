from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from imgattck.config import ModelConfig


@dataclass
class ModelBundle:
    model: Any
    processor: Any
    tokenizer: Any
    device: torch.device


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(model_config: ModelConfig) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_config.name,
        local_files_only=model_config.local_files_only,
        trust_remote_code=model_config.trust_remote_code,
    )


def load_model_bundle(model_config: ModelConfig) -> ModelBundle:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = resolve_device(model_config.device)
    processor = AutoProcessor.from_pretrained(
        model_config.name,
        local_files_only=model_config.local_files_only,
        trust_remote_code=model_config.trust_remote_code,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = load_tokenizer(model_config)

    kwargs: dict[str, Any] = {
        "local_files_only": model_config.local_files_only,
        "trust_remote_code": model_config.trust_remote_code,
    }
    if model_config.dtype:
        kwargs["dtype"] = model_config.dtype
    if model_config.device_map:
        kwargs["device_map"] = model_config.device_map

    model = AutoModelForImageTextToText.from_pretrained(model_config.name, **kwargs)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not model_config.device_map:
        model.to(device)
    return ModelBundle(model=model, processor=processor, tokenizer=tokenizer, device=device)


def reset_rope_cache(model: Any) -> None:
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None


def next_token_logits(model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    reset_rope_cache(model)
    outputs = model(**batch, logits_to_keep=1, use_cache=False, return_dict=True)
    return outputs.logits[:, -1, :]


def image_features(model: Any, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
    output = model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, return_dict=True)
    pooler_output = output.pooler_output
    if isinstance(pooler_output, (tuple, list)):
        return torch.cat(list(pooler_output), dim=0)
    return pooler_output


def next_token_logits_from_image_embeds(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_embeds: torch.Tensor,
) -> torch.Tensor:
    reset_rope_cache(model)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_token = int(model.config.image_token_id)
    image_mask = (input_ids == image_token).unsqueeze(-1).expand_as(inputs_embeds)
    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    position_ids = model.model.compute_3d_position_ids(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
    )
    outputs = model.model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    )
    logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])
    return logits[:, -1, :]
