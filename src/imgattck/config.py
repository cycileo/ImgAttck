from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3.5-4B"
    dtype: str = "auto"
    device: str = "auto"
    device_map: str | None = None
    local_files_only: bool = False
    trust_remote_code: bool = False


@dataclass
class PromptConfig:
    text: str = "Describe the image."
    add_generation_prompt: bool = True


@dataclass
class ImageConfig:
    height: int = 224
    width: int = 224
    patch_size: int = 16
    temporal_patch_size: int = 2
    merge_size: int = 2
    mean: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    std: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])


@dataclass
class OptimizationConfig:
    steps: int = 100
    lr: float = 0.05
    seed: int = 0
    log_every: int = 10
    init: str = "gray"
    init_image: str | None = None
    tv_weight: float = 0.0
    l2_weight: float = 0.0
    latent_match_weight: float = 1.0


@dataclass
class OutputConfig:
    root: str = "runs"
    name: str | None = None


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    target_strings: list[str] = field(default_factory=lambda: [" yes", " no"])


T = TypeVar("T")


def load_config(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return _from_dict(ExperimentConfig, data)


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(to_dict(config), sort_keys=False))


def to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    field_map = {field.name: field for field in fields(cls)}
    for name, field_info in field_map.items():
        if name not in data:
            continue
        value = data[name]
        default_value = _default_for(field_info)
        if is_dataclass(default_value) and isinstance(value, dict):
            kwargs[name] = _from_dict(type(default_value), value)
        else:
            kwargs[name] = value

    unknown = sorted(set(data) - set(field_map))
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {', '.join(unknown)}")
    return cls(**kwargs)


def _default_for(field_info: Any) -> Any:
    if field_info.default is not MISSING:
        return field_info.default
    if field_info.default_factory is not MISSING:
        return field_info.default_factory()  # type: ignore[misc]
    return None
