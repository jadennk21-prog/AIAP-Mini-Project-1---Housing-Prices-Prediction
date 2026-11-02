"""Tests for the command-line entry point (main.py): settings resolution, versioned outputs, run records."""

import json
from pathlib import Path

import pytest

import main
from src.experiment import load_config, save_config

DATA = Path(main.ROOT) / "Housing_Price_Dataset.csv"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="real dataset not available")


def dirs(tmp_path):
    return [
        "--output-dir",
        str(tmp_path / "processed"),
        "--artifact-dir",
        str(tmp_path / "artifacts"),
        "--reports-dir",
        str(tmp_path / "reports"),
    ]


def settings_from(argv):
    return main.resolve_settings(main.parse_args(argv))


# --------------------------------------------------------------------------- #
# Settings: defaults < config file < command line
# --------------------------------------------------------------------------- #
def test_defaults_when_nothing_is_given():
    s = settings_from([])
    assert s.preprocessing.scaler == "standard" and s.selection == "vif" and s.cv_folds == 5 and s.cv_repeats == 3
    assert s.feature_engineering.clip_z == 3.0 and s.data == Path(main.ROOT) / "Housing_Price_Dataset.csv"


def test_config_file_overrides_defaults_and_cli_overrides_file(tmp_path):
    cfg_path = save_config(
        {
            "preprocessing": {"scaler": "robust", "outlier_iqr_factor": 2.0},
            "feature_engineering": {"add_tiers": False},
            "selection": {"method": "lasso"},
            "cv": {"folds": 3, "repeats": 1},
        },
        tmp_path / "e.yaml",
    )
    from_file = settings_from(["--config", str(cfg_path)])
    assert from_file.preprocessing.scaler == "robust" and from_file.preprocessing.outlier_iqr_factor == 2.0
    assert from_file.feature_engineering.add_tiers is False and from_file.selection == "lasso"
    assert (from_file.cv_folds, from_file.cv_repeats) == (3, 1)
    overridden = settings_from(
        ["--config", str(cfg_path), "--scaler", "minmax", "--selection-method", "rfe", "--cv-folds", "4"]
    )
    assert overridden.preprocessing.scaler == "minmax" and overridden.selection == "rfe" and overridden.cv_folds == 4
    assert (
        overridden.preprocessing.outlier_iqr_factor == 2.0 and overridden.cv_repeats == 1
    )  # untouched file values stay


def test_clip_z_zero_turns_winsorising_off():
    assert settings_from(["--clip-z", "0"]).feature_engineering.clip_z is None
    assert settings_from(["--clip-z", "2.5"]).feature_engineering.clip_z == 2.5


def test_boolean_flag_only_overrides_when_given(tmp_path):
    cfg_path = save_config({"preprocessing": {"add_missing_indicator": True}}, tmp_path / "e.yaml")
    assert settings_from(["--config", str(cfg_path)]).preprocessing.add_missing_indicator is True
    assert settings_from([]).preprocessing.add_missing_indicator is False


def test_unknown_config_key_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("preprocessing:\n  scalar: standard\n")
    with pytest.raises(ValueError, match="Unknown preprocessing"):
        settings_from(["--config", str(bad)])


def test_resolved_config_round_trips_through_a_file(tmp_path):
    s = settings_from(["--scaler", "robust", "--categorical-encoding", "auto", "--cv-repeats", "2"])
    path = save_config(s.as_config(), tmp_path / "resolved.yaml")
    again = settings_from(["--config", str(path)])
    assert again.preprocessing == s.preprocessing and again.feature_engineering == s.feature_engineering
    assert again.selection == s.selection and again.cv_repeats == 2


def test_example_config_files_are_valid():
    for name in ("default.yaml", "example_advanced.yaml"):
        path = Path(main.ROOT) / "configs" / name
        if path.exists():
            settings_from(["--config", str(path)])


# --------------------------------------------------------------------------- #
# End-to-end runs
# --------------------------------------------------------------------------- #
@needs_data
def test_save_config_writes_the_resolved_config_and_exits(tmp_path, capsys):
    target = tmp_path / "out" / "mine.yaml"
    assert main.main(["--save-config", str(target), "--scaler", "robust"]) == 0
    assert load_config(target)["preprocessing"]["scaler"] == "robust"


@needs_data
def test_no_save_creates_no_folders(tmp_path):
    assert main.main(["--no-save", *dirs(tmp_path)]) == 0
    assert not (tmp_path / "processed").exists() and not (tmp_path / "reports").exists()


@needs_data
def test_versioned_run_writes_record_config_and_registry(tmp_path):
    code = main.main(["--features", "--run-label", "unit", *dirs(tmp_path)])
    assert code == 0
    run_id = (tmp_path / "reports" / "latest.txt").read_text().strip()
    assert run_id.endswith("-unit")
    report_dir = tmp_path / "reports" / run_id
    for name in ("results.json", "config.yaml", "feature_catalog.csv", "feature_selection_report.csv"):
        assert (report_dir / name).exists(), name
    assert (tmp_path / "processed" / run_id / "features_X_train.csv").exists()
    assert (tmp_path / "artifacts" / run_id / "feature_pipeline.joblib").exists()
    assert (tmp_path / "artifacts" / "latest.txt").read_text().strip() == run_id

    record = json.loads((report_dir / "results.json").read_text())
    assert record["run_id"] == run_id and record["data"]["rows"] == 1000 and len(record["data"]["sha256"]) == 64
    assert record["features"]["engineered"] == 48 and record["preprocessing"]["train_shape"] == [800, 9]
    stages = {s["stage"] for s in record["timings"]["stages"]}
    assert {"load data", "preprocessing", "feature engineering + selection"} <= stages
    registry = (tmp_path / "reports" / "runs.csv").read_text()
    assert run_id in registry and "engineered_features" in registry


@needs_data
def test_rerunning_from_the_saved_config_reproduces_the_same_config_hash(tmp_path):
    main.main(["--features", "--scaler", "robust", "--selection-method", "lasso", *dirs(tmp_path)])
    run_id = (tmp_path / "reports" / "latest.txt").read_text().strip()
    saved = tmp_path / "reports" / run_id / "config.yaml"
    main.main(["--config", str(saved), "--features", *dirs(tmp_path)])
    second = (tmp_path / "reports" / "latest.txt").read_text().strip()
    assert second != run_id  # a new folder, even within the same second
    first_record = json.loads((tmp_path / "reports" / run_id / "results.json").read_text())
    second_record = json.loads((tmp_path / "reports" / second / "results.json").read_text())
    assert first_record["config_hash"] == second_record["config_hash"]
    assert first_record["features"]["kept"] == second_record["features"]["kept"]
    assert second_record["config"]["preprocessing"]["scaler"] == "robust"


@needs_data
def test_two_runs_do_not_overwrite_each_other(tmp_path):
    main.main(["--run-label", "a", *dirs(tmp_path)])
    main.main(["--run-label", "b", *dirs(tmp_path)])
    run_dirs = [p for p in (tmp_path / "reports").iterdir() if p.is_dir()]
    assert len(run_dirs) == 2
    with open(tmp_path / "reports" / "runs.csv") as handle:
        assert len(handle.read().strip().splitlines()) == 3  # header + two runs


@needs_data
def test_identical_runs_in_the_same_second_get_distinct_ids(tmp_path):
    for _ in range(3):
        main.main(["--run-label", "same", *dirs(tmp_path)])
    run_dirs = sorted(p.name for p in (tmp_path / "reports").iterdir() if p.is_dir())
    assert len(run_dirs) == 3 and len(set(run_dirs)) == 3
    assert any(name.endswith("-2") for name in run_dirs) or len({n.rsplit("-", 1)[0] for n in run_dirs}) > 1


@needs_data
def test_no_versioning_uses_the_flat_layout(tmp_path):
    assert main.main(["--features", "--no-versioning", *dirs(tmp_path)]) == 0
    assert (tmp_path / "processed" / "X_train.csv").exists()
    assert (tmp_path / "reports" / "results.json").exists() and (tmp_path / "reports" / "config.yaml").exists()
    assert not (tmp_path / "reports" / "latest.txt").exists()


@needs_data
def test_profile_flag_records_memory(tmp_path):
    assert main.main(["--profile", "--no-save"]) == 0


@needs_data
def test_unknown_selection_method_is_reported(tmp_path):
    with pytest.raises(SystemExit, match="Unknown selection method"):
        main.main(["--features", "--selection-method", "magic", "--no-save"])


@needs_data
def test_chained_selection_from_the_command_line(tmp_path, capsys):
    assert main.main(["--features", "--selection-method", "vif+lasso", "--no-save"]) == 0
    assert "Selection 'vif+lasso'" in capsys.readouterr().out


@needs_data
def test_high_cardinality_options_run_end_to_end(tmp_path, capsys):
    assert (
        main.main(["--categorical-encoding", "auto", "--numeric-imputer", "knn", "--missing-indicator", "--no-save"])
        == 0
    )
    out = capsys.readouterr().out
    assert "Categorical encoding" in out and "location_type" in out


@needs_data
def test_validation_run_records_checks(tmp_path):
    assert main.main(["--validate", *dirs(tmp_path)]) == 0
    run_id = (tmp_path / "reports" / "latest.txt").read_text().strip()
    record = json.loads((tmp_path / "reports" / run_id / "results.json").read_text())
    assert record["validation"]["passed"] is True and len(record["validation"]["checks"]) == 17


@needs_data
def test_baseline_run_writes_tables_plot_model_and_record(tmp_path, capsys):
    assert main.main(["--baseline", *dirs(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "BASELINE BENCHMARK" in out and "Error grows with price" in out
    run_id = (tmp_path / "reports" / "latest.txt").read_text().strip()
    report_dir = tmp_path / "reports" / run_id
    for name in (
        "baseline_cv_comparison.csv",
        "baseline_holdout_metrics.csv",
        "baseline_holdout_predictions.csv",
        "baseline_residuals_by_price_band.csv",
        "baseline_residuals_by_group.csv",
        "baseline_diagnostics.png",
    ):
        assert (report_dir / name).exists(), name
    assert (tmp_path / "artifacts" / run_id / "baseline_model.joblib").exists()
    record = json.loads((report_dir / "results.json").read_text())
    assert record["baseline"]["reference"] == "linear regression + preprocessing (log target)"
    assert len(record["baseline"]["fold_metrics"]["mean price (ignores all features)"]["rmse"]) == 15
    registry = (tmp_path / "reports" / "runs.csv").read_text()
    assert "baseline_holdout_rmse" in registry and "mean_predictor_holdout_rmse" in registry


@needs_data
def test_baseline_no_save_and_candidates(tmp_path, capsys):
    assert main.main(["--baseline", "--candidates", "--cv-folds", "3", "--cv-repeats", "1", "--no-save"]) == 0
    out = capsys.readouterr().out
    assert "random forest + preprocessing" in out and "engineered features (VIF)" in out
    assert not (tmp_path / "reports").exists()


@needs_data
def test_baseline_results_are_the_documented_numbers(capsys):
    """Guards the figures quoted in the README (seed 42, 5-fold x 3 repeats, 800 / 200 split)."""
    assert main.main(["--baseline", "--no-save"]) == 0
    out = capsys.readouterr().out
    assert "29191.0" in out or "29191" in out  # reference hold-out RMSE
    assert "143441" in out  # mean predictor hold-out RMSE


# --------------------------------------------------------------------------- #
# Logging, validation split and reload checks in the command line
# --------------------------------------------------------------------------- #
def test_main_py_never_calls_print():
    import ast

    tree = ast.parse((Path(main.ROOT) / "main.py").read_text())
    prints = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "print"]
    assert prints == []


@needs_data
def test_quiet_silences_the_program_output_but_still_writes_files(tmp_path, capsys):
    assert main.main(["--quiet", "--baseline", *dirs(tmp_path)]) == 0
    assert capsys.readouterr().out == ""
    run_id = (tmp_path / "reports" / "latest.txt").read_text().strip()
    assert (tmp_path / "reports" / run_id / "results.json").exists()


@needs_data
def test_default_output_goes_to_stdout_through_the_logger(capsys):
    assert main.main(["--no-save"]) == 0
    out = capsys.readouterr().out
    assert "Loaded 1000 rows x 9 columns" in out and "Stage timings" in out


@needs_data
def test_repeated_runs_do_not_duplicate_output_or_leak_handlers(capsys):
    import logging

    main.main(["--no-save"])
    first = capsys.readouterr().out
    main.main(["--no-save"])
    second = capsys.readouterr().out
    assert first.count("Loaded 1000 rows") == 1 and second.count("Loaded 1000 rows") == 1
    assert not any(getattr(h, main._HANDLER_TAG, False) for h in logging.getLogger().handlers + main.logger.handlers)


@needs_data
def test_run_log_is_written_with_program_output_and_library_messages(tmp_path):
    main.main(["--baseline", *dirs(tmp_path)])
    run_id = (tmp_path / "reports" / "latest.txt").read_text().strip()
    log = (tmp_path / "reports" / run_id / "run.log").read_text()
    assert "main: Run ID:" in log and "main: Loaded 1000 rows x 9 columns" in log
    assert "src.preprocessing: Loaded 1000 rows" in log and "src.experiment: Stage 'preprocessing' took" in log


@needs_data
def test_library_messages_appear_on_stderr_only_with_verbose(capsys):
    main.main(["--no-save"])
    assert "INFO:" not in capsys.readouterr().err
    main.main(["--no-save", "--verbose"])
    assert "INFO: Loaded 1000 rows" in capsys.readouterr().err


@needs_data
def test_validation_size_gives_a_70_10_20_split_end_to_end(tmp_path, capsys):
    assert main.main(["--baseline", "--validation-size", "0.1", *dirs(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Train: (700, 9), Test: (200, 9)" in out and "Validation set (100 rows" in out
    run_id = (tmp_path / "reports" / "latest.txt").read_text().strip()
    record = json.loads((tmp_path / "reports" / run_id / "results.json").read_text())
    assert record["baseline"]["protocol"]["n_validation"] == 100 and record["baseline"]["validation"] is not None
    assert record["config"]["preprocessing"]["validation_size"] == 0.1
    assert (tmp_path / "reports" / run_id / "baseline_validation_metrics.csv").exists()
    assert (tmp_path / "processed" / run_id / "X_train.csv").read_text().count("\n") == 701


@needs_data
def test_saved_artifacts_are_reloaded_and_verified(tmp_path, capsys):
    assert main.main(["--baseline", "--features", *dirs(tmp_path)]) == 0
    out = capsys.readouterr().out
    for name in ("preprocessor.joblib", "baseline_model.joblib", "feature_pipeline.joblib"):
        assert f"Reload check passed: {name}" in out
    run_id = (tmp_path / "reports" / "latest.txt").read_text().strip()
    record = json.loads((tmp_path / "reports" / run_id / "results.json").read_text())
    checks = {Path(c["path"]).name: c for c in record["artifact_checks"]}
    assert set(checks) == {"preprocessor.joblib", "baseline_model.joblib", "feature_pipeline.joblib"}
    assert all(c["ok"] and c["n_rows"] == 5 for c in checks.values())


@needs_data
def test_a_failing_reload_check_stops_the_run(tmp_path, monkeypatch):
    from src import experiment

    def corrupt_then_verify(path, sample, **kwargs):
        Path(path).write_bytes(b"corrupted")
        return experiment.verify_saved_artifact(path, sample, **kwargs)

    monkeypatch.setattr(main, "verify_saved_artifact", corrupt_then_verify)
    with pytest.raises(experiment.ArtifactVerificationError):
        main.main(["--baseline", *dirs(tmp_path)])
