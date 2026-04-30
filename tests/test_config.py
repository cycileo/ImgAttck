from imgattck.config import experiment_model_configs, load_config


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
