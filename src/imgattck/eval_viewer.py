from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def discover_eval_runs(root: str | Path = "runs") -> list[Path]:
    run_root = Path(root).expanduser()
    return sorted(path for path in run_root.glob("eval-*") if (path / "results.json").exists())


def latest_eval_run(root: str | Path = "runs") -> Path | None:
    runs = discover_eval_runs(root)
    return runs[-1] if runs else None


def load_eval_run(run_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    run_path = Path(run_dir).expanduser()
    results = _read_json(run_path / "results.json")
    summary_path = run_path / "summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else None
    return results, summary


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="ImgAttck Evaluation", layout="wide")
    st.title("ImgAttck Evaluation")

    root_text = st.sidebar.text_input("Runs root", value="runs")
    runs = discover_eval_runs(root_text)
    if not runs:
        st.warning(f"No eval runs found under {Path(root_text).expanduser()}.")
        return

    labels = [str(path) for path in runs]
    selected = st.sidebar.selectbox("Eval run", labels, index=len(labels) - 1)
    run_dir = Path(selected)
    results, summary = load_eval_run(run_dir)

    image_path = _resolve_image_path(results.get("image"))
    image_col, summary_col = st.columns([1, 2])
    with image_col:
        st.subheader("Image")
        if image_path and image_path.exists():
            st.image(str(image_path), use_container_width=True)
        elif image_path:
            st.info(f"Image not found: {image_path}")
        else:
            st.info("This run did not record an image path.")
        st.caption(str(image_path) if image_path else "No image")

    with summary_col:
        st.subheader("Run")
        st.write(str(run_dir))
        _render_summary(st, summary)

    st.subheader("Model Results")
    for model_result in results.get("models", []):
        model_name = str(model_result.get("model", "unknown model"))
        answers = model_result.get("answers", [])
        label = f"{model_result.get('model_index', '?')}: {model_name}"
        with st.expander(label, expanded=True):
            if not answers:
                st.info("No answers recorded for this model.")
                continue
            for answer in answers:
                _render_answer(st, answer)


def _render_summary(st: Any, summary: dict[str, Any] | None) -> None:
    if not summary:
        st.info("No summary.json was found for this run.")
        return
    cols = st.columns(4)
    cols[0].metric("Models", summary.get("total_models", "-"))
    cols[1].metric("Questions", summary.get("total_questions", "-"))
    cols[2].metric("Trials", summary.get("total_trials", "-"))
    cols[3].metric("Success rate", _format_percent(summary.get("success_rate")))

    by_model = summary.get("by_model", [])
    if by_model:
        st.caption("Success is measured on the image-conditioned answer.")
        st.dataframe(by_model, hide_index=True, use_container_width=True)


def _render_answer(st: Any, answer: dict[str, Any]) -> None:
    question_index = answer.get("question_index", "?")
    st.markdown(f"**Question {question_index}**")
    st.write(answer.get("question", ""))

    without_image = answer.get("without_image", {})
    with_image = answer.get("with_image", {})
    answer_cols = st.columns(2)
    with answer_cols[0]:
        st.caption("Without image")
        st.write(without_image.get("answer", answer.get("answer_without_image", "")))
        st.metric("Target probability", _format_float(without_image.get("target_probability")))
        st.write(f"Success: {_format_bool(without_image.get('success'))}")
    with answer_cols[1]:
        st.caption("With image")
        st.write(with_image.get("answer", answer.get("answer", "")))
        st.metric(
            "Target probability",
            _format_float(with_image.get("target_probability")),
            delta=_format_delta(answer.get("target_probability_delta")),
        )
        st.write(f"Success: {_format_bool(with_image.get('success', answer.get('success')))}")

    rows = _display_token_rows(answer.get("token_comparison", []))
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info("No token comparison rows were recorded for this answer.")
    st.divider()


def _display_token_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    display_rows: list[dict[str, Any]] = []
    for row in rows:
        display_rows.append(
            {
                "target": row.get("text"),
                "token_id": row.get("token_id"),
                "decoded": row.get("decoded"),
                "logit without": row.get("logit_without_image"),
                "logit with": row.get("logit_with_image"),
                "logit delta": row.get("logit_delta"),
                "prob without": row.get("probability_without_image"),
                "prob with": row.get("probability_with_image"),
                "prob delta": row.get("probability_delta"),
                "prob ratio": row.get("probability_ratio"),
            }
        )
    return display_rows


def _resolve_image_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _format_delta(value: Any) -> str | None:
    if value is None:
        return None
    return _format_float(value)


def _format_percent(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _format_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "-"


if __name__ == "__main__":
    main()
