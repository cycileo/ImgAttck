from imgattck.eval_viewer import discover_eval_runs, latest_eval_run, load_eval_run


def test_discover_eval_runs_uses_latest_eval_with_results(tmp_path):
    older = tmp_path / "runs" / "eval-20260430-120000"
    newer = tmp_path / "runs" / "eval-20260430-130000"
    incomplete = tmp_path / "runs" / "eval-20260430-140000"
    pixel = tmp_path / "runs" / "pixel-20260430-150000"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    pixel.mkdir(parents=True)
    (older / "results.json").write_text('{"models": []}')
    (newer / "results.json").write_text('{"models": [{"model": "new"}]}')

    runs = discover_eval_runs(tmp_path / "runs")

    assert runs == [older, newer]
    assert latest_eval_run(tmp_path / "runs") == newer


def test_load_eval_run_reads_optional_summary(tmp_path):
    run = tmp_path / "eval-20260430-120000"
    run.mkdir()
    (run / "results.json").write_text('{"image": "optimized.png", "models": []}')
    (run / "summary.json").write_text('{"success_rate": 0.5}')

    results, summary = load_eval_run(run)

    assert results["image"] == "optimized.png"
    assert summary == {"success_rate": 0.5}
