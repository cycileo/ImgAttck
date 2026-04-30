from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from imgattck.preprocess import QwenImageSpec


def make_pil_image(spec: QwenImageSpec, value: int = 127) -> Image.Image:
    return Image.new("RGB", (spec.width, spec.height), color=(value, value, value))


def load_rgb_tensor(path: str | Path, spec: QwenImageSpec | None = None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if spec is not None:
        image = image.resize((spec.width, spec.height), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def save_rgb_tensor(image: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 4:
        image = image[0]
    image = image.detach().clamp(0.0, 1.0).cpu()
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def initial_image_tensor(kind: str, spec: QwenImageSpec, seed: int, init_image: str | None = None) -> torch.Tensor:
    if init_image:
        return load_rgb_tensor(init_image, spec).unsqueeze(0)
    if kind == "gray":
        return torch.full((1, 3, spec.height, spec.width), 0.5)
    if kind == "black":
        return torch.zeros((1, 3, spec.height, spec.width))
    if kind == "white":
        return torch.ones((1, 3, spec.height, spec.width))
    if kind == "noise":
        generator = torch.Generator().manual_seed(seed)
        return torch.rand((1, 3, spec.height, spec.width), generator=generator)
    raise ValueError(f"Unknown image initialization {kind!r}; use gray, black, white, noise, or init_image.")
