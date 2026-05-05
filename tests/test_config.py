from imgattck.config import experiment_model_configs, experiment_prompt_configs, load_config


def test_load_config_accepts_multiple_models(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
models:
  - name: first
    device: cpu
  - second
optimization:
  early_stop_loss: 0.25
"""
    )

    config = load_config(config_path)
    models = experiment_model_configs(config)

    assert [model.name for model in models] == ["first", "second"]
    assert models[0].device == "cpu"
    assert config.optimization.early_stop_loss == 0.25


def test_load_config_falls_back_to_single_model(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: only
"""
    )

    config = load_config(config_path)

    assert [model.name for model in experiment_model_configs(config)] == ["only"]


def test_load_config_falls_back_to_single_prompt(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
prompt:
  text: "Describe this."
  enable_thinking: false
"""
    )

    config = load_config(config_path)
    prompts = experiment_prompt_configs(config)

    assert len(prompts) == 1
    assert prompts[0].text == "Describe this."
    assert prompts[0].enable_thinking is False


def test_load_config_accepts_multiple_prompts_with_shared_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
prompt:
  add_generation_prompt: false
  enable_thinking: false
prompts:
  - "First prompt."
  - text: "Second prompt."
    enable_thinking: true
"""
    )

    config = load_config(config_path)
    prompts = experiment_prompt_configs(config)

    assert [prompt.text for prompt in prompts] == ["First prompt.", "Second prompt."]
    assert [prompt.add_generation_prompt for prompt in prompts] == [False, False]
    assert [prompt.enable_thinking for prompt in prompts] == [False, True]
