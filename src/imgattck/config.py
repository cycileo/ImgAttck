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
    enable_thinking: bool = True


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
class GenerationConfig:
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    num_beams: int | None = None
    skip_special_tokens: bool = True


@dataclass
class SuccessConfig:
    strings: list[str] = field(default_factory=lambda: ["yes"])
    case_sensitive: bool = False
    mode: str = "contains"
    strip: bool = True


@dataclass
class EvaluationQuestion:
    text: str
    success_strings: list[str] | None = None


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    target_strings: list[str] = field(default_factory=lambda: [" yes", " no"])


@dataclass
class EvaluationConfig:
    models: list[ModelConfig] = field(default_factory=lambda: [ModelConfig()])
    image: str = "runs/<run>/optimized.png"
    questions: list[EvaluationQuestion] = field(default_factory=list)
    image_spec: ImageConfig = field(default_factory=ImageConfig)
    prompt: PromptConfig = field(default_factory=lambda: PromptConfig(text="", enable_thinking=False))
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    success: SuccessConfig = field(default_factory=SuccessConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


T = TypeVar("T")


def load_config(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return _from_dict(ExperimentConfig, data)


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")

    field_map = {field.name for field in fields(EvaluationConfig)}
    unknown = sorted(set(data) - field_map)
    if unknown:
        raise ValueError(f"Unknown config keys for EvaluationConfig: {', '.join(unknown)}")

    models_data = _as_list(data.get("models", [ModelConfig().name]), "models")
    models = [_parse_model_config(item) for item in models_data]
    if not models:
        raise ValueError("Evaluation config must contain at least one model.")
    questions_data = _as_list(data.get("questions", []), "questions")
    questions = [_parse_evaluation_question(item) for item in questions_data]
    if not questions:
        raise ValueError("Evaluation config must contain at least one question.")
    prompt_data = {"text": "", "add_generation_prompt": True, "enable_thinking": False}
    prompt_data.update(data.get("prompt", {}))

    return EvaluationConfig(
        models=models,
        image=data.get("image", EvaluationConfig.image),
        questions=questions,
        image_spec=_from_dict(ImageConfig, data.get("image_spec", {})),
        prompt=_from_dict(PromptConfig, prompt_data),
        generation=_from_dict(GenerationConfig, data.get("generation", {})),
        success=_from_dict(SuccessConfig, data.get("success", {})),
        output=_from_dict(OutputConfig, data.get("output", {})),
    )


def dump_config(config: Any, path: str | Path) -> None:
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


def _parse_model_config(value: Any) -> ModelConfig:
    if isinstance(value, str):
        return ModelConfig(name=value)
    if isinstance(value, dict):
        return _from_dict(ModelConfig, value)
    raise ValueError(f"Model entries must be strings or mappings, got {type(value).__name__}.")


def _parse_evaluation_question(value: Any) -> EvaluationQuestion:
    if isinstance(value, str):
        return EvaluationQuestion(text=value)
    if isinstance(value, dict):
        return _from_dict(EvaluationQuestion, value)
    raise ValueError(f"Question entries must be strings or mappings, got {type(value).__name__}.")


def _as_list(value: Any, field_name: str) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)):
        return [value]
    raise ValueError(f"{field_name} must be a list, string, or mapping; got {type(value).__name__}.")


def _default_for(field_info: Any) -> Any:
    if field_info.default is not MISSING:
        return field_info.default
    if field_info.default_factory is not MISSING:
        return field_info.default_factory()  # type: ignore[misc]
    return None
