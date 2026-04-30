import pytest
import torch

from imgattck.config import SuccessConfig, load_evaluation_config
from imgattck.evaluation import answer_matches, compare_target_tokens, resolve_evaluation_image_path, summarize_results


def test_load_evaluation_config_accepts_model_and_question_shortcuts(tmp_path):
    path = tmp_path / "evaluate.yaml"
    path.write_text(
        """
models:
  - Qwen/Qwen3.5-4B
image: runs/example/optimized.png
questions:
  - "Describe the image."
  - text: "Answer yes or no."
    success_strings: ["yes"]
    target_strings: [" yes"]
target_strings: [" no"]
success:
  strings: ["YES"]
""".lstrip()
    )

    config = load_evaluation_config(path)

    assert config.models[0].name == "Qwen/Qwen3.5-4B"
    assert config.image == "runs/example/optimized.png"
    assert config.questions[0].text == "Describe the image."
    assert config.questions[1].success_strings == ["yes"]
    assert config.questions[1].target_strings == [" yes"]
    assert config.target_strings == [" no"]
    assert config.success.strings == ["YES"]
    assert config.prompt.enable_thinking is False


def test_load_evaluation_config_allows_omitted_image(tmp_path):
    path = tmp_path / "evaluate.yaml"
    path.write_text(
        """
models:
  - Qwen/Qwen3.5-4B
questions:
  - "Describe the image."
""".lstrip()
    )

    config = load_evaluation_config(path)

    assert config.image is None


def test_resolve_evaluation_image_path_uses_latest_pixel_run(tmp_path):
    older = tmp_path / "runs" / "pixel-20260430-120000"
    newer = tmp_path / "runs" / "pixel-20260430-130000"
    ignored = tmp_path / "runs" / "native-20260430-140000"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    ignored.mkdir(parents=True)
    (older / "optimized.png").write_text("older")
    (newer / "optimized.png").write_text("newer")
    (ignored / "optimized.png").write_text("ignored")

    path = tmp_path / "evaluate.yaml"
    path.write_text(
        f"""
models:
  - Qwen/Qwen3.5-4B
questions:
  - "Describe the image."
output:
  root: {tmp_path / "runs"}
""".lstrip()
    )
    config = load_evaluation_config(path)

    assert resolve_evaluation_image_path(config) == newer / "optimized.png"


def test_answer_matches_supports_case_insensitive_prefix():
    config = SuccessConfig(strings=["yes"], case_sensitive=False, mode="prefix", strip=True)

    assert answer_matches(" Yes, that is correct.", config.strings, config)
    assert not answer_matches("I would say no.", config.strings, config)


def test_summarize_results_counts_by_model_and_question():
    rows = [
        {"model_index": 0, "model": "model-a", "question_index": 0, "question": "q1", "success": True},
        {"model_index": 0, "model": "model-a", "question_index": 1, "question": "q2", "success": False},
        {"model_index": 1, "model": "model-b", "question_index": 0, "question": "q1", "success": True},
    ]

    summary = summarize_results(rows, total_models=2, total_questions=2)

    assert summary["successes"] == 2
    assert summary["total_trials"] == 3
    assert summary["by_model"][0]["successes"] == 1
    assert summary["by_model"][1]["successes"] == 1
    assert summary["by_question"][0]["successes"] == 2


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        mapping = {
            " yes": [1],
            " no": [2],
            "multi": [1, 2],
        }
        return mapping[text]

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        reverse = {
            1: " yes",
            2: " no",
        }
        return "".join(reverse[token_id] for token_id in token_ids)


def test_compare_target_tokens_reports_before_after_metrics():
    without_image_logits = torch.zeros((1, 4))
    with_image_logits = torch.tensor([[0.0, 2.0, -1.0, 0.0]])

    result = compare_target_tokens(
        FakeTokenizer(),
        without_image_logits,
        with_image_logits,
        [" yes", " no"],
    )

    assert result["target_probability_without_image"] == pytest.approx(0.5)
    assert result["target_probability_with_image"] > result["target_probability_without_image"]
    yes_row = result["token_comparison"][0]
    assert yes_row["text"] == " yes"
    assert yes_row["token_id"] == 1
    assert yes_row["logit_delta"] == pytest.approx(2.0)
    assert yes_row["probability_with_image"] > yes_row["probability_without_image"]


def test_compare_target_tokens_rejects_multi_token_target():
    with pytest.raises(ValueError, match="exactly one token"):
        compare_target_tokens(FakeTokenizer(), torch.zeros((1, 4)), torch.zeros((1, 4)), ["multi"])
