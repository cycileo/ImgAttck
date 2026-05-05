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
uv run imgattck evaluate-image configs/evaluate.yaml
uv run streamlit run src/imgattck/eval_viewer.py
```

Each optimization run writes a config snapshot, token report, metrics JSON/CSV,
the final image or latent, and native validation results where applicable.
Evaluation runs write generated answers plus success summaries to
`results.json`, `results.csv`, and `summary.json`.

## Config

Start from [configs/default.yaml](configs/default.yaml). Important knobs:

- `target_strings`: strings whose next-token probability should increase. Each
  must encode to exactly one tokenizer token.
- `prompts`: optional list of prompt strings or prompt mappings. If omitted,
  the single `prompt` block is used as before. Shared settings from `prompt`
  are inherited by string entries and can be overridden per prompt.
- `models`: one or more model configs. Pixel optimization averages the
  target-token loss across all listed models and prompts before each shared
  image update.
- `prompt.enable_thinking`: keep `true` to optimize the first reasoning token,
  or set `false` to use Qwen's empty `<think></think>` block and optimize the
  first answer token.
- `image`: fixed differentiable preprocessing shape. Keep dimensions divisible
  by `patch_size * merge_size`. Native validation passes matching
  `min_pixels`/`max_pixels` to the official processor so it uses the same grid.
- `optimization`: steps, learning rate, `early_stop_loss`, initialization, and
  optional TV/L2 image regularizers.
- `models[].device` / `models[].device_map`: use a GPU-capable environment for full
  4B optimization.

For multiple optimization models:

```yaml
models:
  - name: Qwen/Qwen3.5-4B
    device: auto
  - name: /path/to/another/local/vlm
    device: auto
```

For multiple optimization prompts:

```yaml
prompt:
  add_generation_prompt: true
  enable_thinking: false

prompts:
  - "Describe the image."
  - text: "Answer the question using only one word."
    enable_thinking: true
```

The optimization objective is averaged across every model/prompt pair. Runtime
therefore scales roughly with `len(models) * len(prompts)`.

Latent optimization and latent inversion currently require exactly one model,
because the optimized visual latent shape is model-specific.

Use [configs/evaluate.yaml](configs/evaluate.yaml) to test an optimized image
across multiple models and questions. The evaluator now runs each question
twice: once without an image and once with the optimized image. It records both
answers, success flags, and the before/after logits and probabilities for the
single-token `target_strings`. Question-level `target_strings` override the
global list; if no target strings are configured, the evaluator falls back to
the success strings. If `image` is omitted in the evaluation config, it
automatically uses `optimized.png` from the newest `output.root/pixel-*` run.

The Streamlit viewer opens the latest `runs/eval-*` directory by default and
shows the image, summary, answers, and target-token comparison tables:

```bash
uv run streamlit run src/imgattck/eval_viewer.py
```

## Tests

```bash
uv run pytest
```
