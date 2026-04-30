from imgattck.config import SuccessConfig, load_evaluation_config
from imgattck.evaluation import answer_matches, summarize_results


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
success:
  strings: ["YES"]
""".lstrip()
    )

    config = load_evaluation_config(path)

    assert config.models[0].name == "Qwen/Qwen3.5-4B"
    assert config.image == "runs/example/optimized.png"
    assert config.questions[0].text == "Describe the image."
    assert config.questions[1].success_strings == ["yes"]
    assert config.success.strings == ["YES"]
    assert config.prompt.enable_thinking is False


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
