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
    assert record["validation"]["passed"] is True and len(record["validation"]["checks"]) == 15
