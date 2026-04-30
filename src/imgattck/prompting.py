from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from imgattck.preprocess import QwenImageSpec


def get_tokenizer(processor_or_tokenizer: object) -> object:
    return getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)


def image_token_id(tokenizer: object) -> int:
    if getattr(tokenizer, "image_token_id", None) is not None:
        return int(getattr(tokenizer, "image_token_id"))
    return int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))  # type: ignore[attr-defined]


def manual_prompt_inputs(
    tokenizer: object,
    prompt: str,
    spec: QwenImageSpec,
    add_generation_prompt: bool = True,
) -> dict[str, torch.Tensor]:
    image_tokens = "<|image_pad|>" * spec.num_image_tokens
    text = (
        "<|im_start|>user\n"
        f"<|vision_start|>{image_tokens}<|vision_end|>{prompt}<|im_end|>\n"
    )
    if add_generation_prompt:
        text += "<|im_start|>assistant\n<think>\n"
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)  # type: ignore[operator]
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[input_ids == image_token_id(tokenizer)] = 1
    image_grid_thw = torch.tensor([spec.grid_thw], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "image_grid_thw": image_grid_thw,
    }


def make_messages(prompt: str, image: Image.Image) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def native_processor_inputs(
    processor: object,
    prompt: str,
    image: Image.Image,
    spec: QwenImageSpec,
    add_generation_prompt: bool = True,
) -> dict[str, torch.Tensor]:
    if not hasattr(processor, "apply_chat_template"):
        raise TypeError("Native validation requires a multimodal processor with apply_chat_template.")
    inputs = processor.apply_chat_template(  # type: ignore[attr-defined]
        make_messages(prompt, image),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "min_pixels": spec.height * spec.width,
            "max_pixels": spec.height * spec.width,
        },
    )
    batch = dict(inputs)
    tokenizer = get_tokenizer(processor)
    if "mm_token_type_ids" not in batch:
        mm_token_type_ids = torch.zeros_like(batch["input_ids"])
        mm_token_type_ids[batch["input_ids"] == image_token_id(tokenizer)] = 1
        batch["mm_token_type_ids"] = mm_token_type_ids
    return batch


def move_batch(batch: dict[str, torch.Tensor], device: torch.device | str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
