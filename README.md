# ImgAttck

Framework for studying how optimized images influence the next-token distribution
of a local vision-language model. The default target is `Qwen/Qwen3.5-4B`.

The main path optimizes pixels end-to-end through a differentiable clone of the
Qwen image preprocessing path. It then validates the resulting PNG with the
native Hugging Face processor/model pipeline. A latent oracle path is also
included for comparing reachable image effects against directly optimized visual
embeddings.

## Setup

```bash
uv sync
```

The project pins `transformers` to GitHub `main` because released/local
Transformers builds may not yet recognize `model_type: qwen3_5`.

## Commands

```bash
uv run imgattck check-tokens configs/default.yaml
uv run imgattck optimize-pixels configs/default.yaml
uv run imgattck validate-native configs/default.yaml runs/<run>/optimized.png
uv run imgattck optimize-latent configs/default.yaml
uv run imgattck invert-latent configs/default.yaml runs/<latent-run>/latent.pt
```

Each optimization run writes a config snapshot, token report, metrics JSON/CSV,
the final image or latent, and native validation results where applicable.

## Config

Start from [configs/default.yaml](configs/default.yaml). Important knobs:

- `target_strings`: strings whose next-token probability should increase. Each
  must encode to exactly one tokenizer token.
- `image`: fixed differentiable preprocessing shape. Keep dimensions divisible
  by `patch_size * merge_size`. Native validation passes matching
  `min_pixels`/`max_pixels` to the official processor so it uses the same grid.
- `optimization`: steps, learning rate, initialization, and optional TV/L2 image
  regularizers.
- `model.device` / `model.device_map`: use a GPU-capable environment for full
  4B optimization.

## Tests

```bash
uv run pytest
```
