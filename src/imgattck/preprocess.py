from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from imgattck.config import ImageConfig


@dataclass(frozen=True)
class QwenImageSpec:
    height: int = 224
    width: int = 224
    patch_size: int = 16
    temporal_patch_size: int = 2
    merge_size: int = 2
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    @property
    def grid_thw(self) -> tuple[int, int, int]:
        return (1, self.height // self.patch_size, self.width // self.patch_size)

    @property
    def num_image_tokens(self) -> int:
        t, h, w = self.grid_thw
        return t * h * w // (self.merge_size**2)


@dataclass
class PreprocessedImage:
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


def spec_from_config(config: ImageConfig) -> QwenImageSpec:
    return QwenImageSpec(
        height=config.height,
        width=config.width,
        patch_size=config.patch_size,
        temporal_patch_size=config.temporal_patch_size,
        merge_size=config.merge_size,
        mean=tuple(config.mean),  # type: ignore[arg-type]
        std=tuple(config.std),  # type: ignore[arg-type]
    )


def validate_spec(spec: QwenImageSpec) -> None:
    factor = spec.patch_size * spec.merge_size
    if spec.height % factor or spec.width % factor:
        raise ValueError(
            f"Image size must be divisible by patch_size * merge_size ({factor}); "
            f"got {spec.height}x{spec.width}."
        )
    if len(spec.mean) != 3 or len(spec.std) != 3:
        raise ValueError("QwenImageSpec mean/std must contain exactly three values.")


def differentiable_qwen_preprocess(image: torch.Tensor, spec: QwenImageSpec) -> PreprocessedImage:
    """Differentiable equivalent of Qwen2VLImageProcessor packing for fixed-size RGB images.

    Input image values must be in [0, 1] with shape [3, H, W] or [B, 3, H, W].
    The output `pixel_values` shape is [sum(grid_t * grid_h * grid_w), C*T*P*P].
    """
    validate_spec(spec)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"Expected image tensor shape [3,H,W] or [B,3,H,W], got {tuple(image.shape)}")

    image = image.clamp(0.0, 1.0)
    if tuple(image.shape[-2:]) != (spec.height, spec.width):
        image = F.interpolate(image, size=(spec.height, spec.width), mode="bicubic", align_corners=False)
        image = image.clamp(0.0, 1.0)

    mean = torch.tensor(spec.mean, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor(spec.std, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    image = (image - mean) / std

    packed = [_pack_single_image(sample, spec) for sample in image]
    pixel_values = torch.cat(packed, dim=0)
    grids = [spec.grid_thw for _ in packed]
    image_grid_thw = torch.tensor(grids, dtype=torch.long, device=image.device)
    return PreprocessedImage(pixel_values=pixel_values, image_grid_thw=image_grid_thw)


def _pack_single_image(image: torch.Tensor, spec: QwenImageSpec) -> torch.Tensor:
    channel = image.shape[0]
    grid_t, grid_h, grid_w = spec.grid_thw
    patch = spec.patch_size
    merge = spec.merge_size
    temporal = spec.temporal_patch_size

    frames = image.unsqueeze(0).repeat(temporal, 1, 1, 1)
    patches = frames.reshape(
        grid_t,
        temporal,
        channel,
        grid_h // merge,
        merge,
        patch,
        grid_w // merge,
        merge,
        patch,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return patches.reshape(grid_t * grid_h * grid_w, channel * temporal * patch * patch)


def image_parameter_to_rgb(parameter: torch.Tensor) -> torch.Tensor:
    return parameter.sigmoid()


def rgb_to_image_parameter(image: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    image = image.clamp(eps, 1.0 - eps)
    return torch.logit(image)


def total_variation(image: torch.Tensor) -> torch.Tensor:
    vertical = (image[..., 1:, :] - image[..., :-1, :]).abs().mean()
    horizontal = (image[..., :, 1:] - image[..., :, :-1]).abs().mean()
    return vertical + horizontal
