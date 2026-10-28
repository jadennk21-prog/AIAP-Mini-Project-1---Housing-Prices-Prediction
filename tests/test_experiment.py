"""Tests for src/experiment.py: config files, run IDs, JSON helpers, profiling and run records."""

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiment import (
    CONFIG_SECTIONS,
    RunRecorder,
    StageProfiler,
    config_hash,
    ensure_unique_run_id,
    environment_info,
    file_sha256,
    load_config,
    make_run_id,
    save_config,
    to_jsonable,
    versioned_dir,
)
from src.feature_engineering import FeatureEngineeringConfig
from src.preprocessing import PreprocessingConfig


@pytest.fixture()
def config():
    return {
        "run": {"data": "Housing_Price_Dataset.csv"},
        "preprocessing": PreprocessingConfig(scaler="robust").to_dict(),
        "feature_engineering": FeatureEngineeringConfig(add_tiers=False).to_dict(),
        "selection": {"method": "vif+rfe"},
        "cv": {"folds": 4, "repeats": 2},
    }


# --------------------------------------------------------------------------- #
# Config files
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".json"])
def test_config_round_trip_in_every_format(config, tmp_path, suffix):
    path = save_config(config, tmp_path / "nested" / f"exp{suffix}")
    loaded = load_config(path)
    assert loaded == to_jsonable(config)
    assert PreprocessingConfig.from_dict(loaded["preprocessing"]) == PreprocessingConfig(scaler="robust")
    assert FeatureEngineeringConfig.from_dict(loaded["feature_engineering"]) == FeatureEngineeringConfig(
        add_tiers=False
    )


def test_yaml_and_json_give_the_same_hash(config, tmp_path):
    a = load_config(save_config(config, tmp_path / "a.yaml"))
    b = load_config(save_config(config, tmp_path / "b.json"))
    assert config_hash(a) == config_hash(b)


def test_load_config_rejects_bad_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
    (tmp_path / "x.txt").write_text("a: 1")
    with pytest.raises(ValueError, match="Unsupported config format"):
        load_config(tmp_path / "x.txt")
    (tmp_path / "bad_section.yaml").write_text("preprocesing:\n  scaler: standard\n")
    with pytest.raises(ValueError, match="Unknown config section"):
        load_config(tmp_path / "bad_section.yaml")
    (tmp_path / "bad_key.yaml").write_text("cv:\n  folds: 5\n  repeat: 3\n")
    with pytest.raises(ValueError, match="Unknown key"):
        load_config(tmp_path / "bad_key.yaml")
    (tmp_path / "list.yaml").write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(tmp_path / "list.yaml")
    (tmp_path / "scalar_section.yaml").write_text("cv: 5\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(tmp_path / "scalar_section.yaml")


def test_empty_config_file_is_allowed(tmp_path):
    (tmp_path / "empty.yaml").write_text("")
    assert load_config(tmp_path / "empty.yaml") == {}


def test_save_config_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        save_config({}, tmp_path / "x.ini")


def test_sections_constant_lists_all_sections():
    assert set(CONFIG_SECTIONS) == {"run", "preprocessing", "feature_engineering", "selection", "cv"}


def test_config_hash_ignores_key_order_but_detects_changes(config):
    reordered = dict(reversed(list(config.items())))
    assert config_hash(config) == config_hash(reordered)
    changed = {**config, "cv": {"folds": 5, "repeats": 2}}
    assert config_hash(config) != config_hash(changed)
    assert len(config_hash(config)) == 8 and len(config_hash(config, length=12)) == 12


# --------------------------------------------------------------------------- #
# Run IDs and folders
# --------------------------------------------------------------------------- #
def test_run_id_format_and_label(config):
    now = datetime(2026, 11, 5, 14, 25, 30, tzinfo=timezone.utc)
    run_id = make_run_id(config, now=now)
    assert run_id == f"20261105-142530-{config_hash(config)}"
    labelled = make_run_id(config, now=now, label="my run/v2!")
    assert labelled == f"{run_id}-my-run-v2-"
    assert make_run_id(config, now=now) == run_id  # same config and time -> same ID


def test_versioned_dir_creates_run_folder_and_latest_pointer(tmp_path):
    first = versioned_dir(tmp_path / "reports", "run-1")
    second = versioned_dir(tmp_path / "reports", "run-2")
    assert first.name == "run-1" and second.name == "run-2" and first.is_dir() and second.is_dir()
    assert (tmp_path / "reports" / "latest.txt").read_text().strip() == "run-2"


def test_versioned_dir_disabled_returns_the_existing_base(tmp_path):
    assert versioned_dir(tmp_path / "out", "run-1", enabled=False) == tmp_path / "out"
    assert (tmp_path / "out").is_dir() and not (tmp_path / "out" / "latest.txt").exists()


def test_file_sha256_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "data.bin"
    path.write_bytes(b"hello world" * 100000)
    assert file_sha256(path) == hashlib.sha256(b"hello world" * 100000).hexdigest()


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
def test_to_jsonable_handles_numpy_pandas_paths_and_nan():
    payload = {
        "int": np.int64(3),
        "float": np.float32(1.5),
        "nan": float("nan"),
        "inf": np.inf,
        "bool": np.bool_(True),
        "array": np.arange(3),
        "tuple": (1, 2),
        "path": Path("a/b"),
        "none": None,
        "series": pd.Series({"x": 1.0, "y": np.nan}),
        "frame": pd.DataFrame({"a": [1, 2]}, index=["r1", "r2"]),
        "when": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    safe = to_jsonable(payload)
    json.dumps(safe)  # must be serialisable
    assert safe["int"] == 3 and safe["nan"] is None and safe["inf"] is None and safe["bool"] is True
    assert safe["array"] == [0, 1, 2] and safe["tuple"] == [1, 2] and safe["path"] == str(Path("a/b"))
    assert safe["series"] == {"x": 1.0, "y": None}
    assert safe["frame"] == {"r1": {"a": 1}, "r2": {"a": 2}}


def test_environment_info_lists_versions():
    info = environment_info()
    assert {"python", "platform", "packages"} <= set(info)
    assert info["packages"]["scikit-learn"] and info["packages"]["pandas"]


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #
def test_profiler_records_stage_durations():
    profiler = StageProfiler()
    with profiler.stage("sleepy"):
        time.sleep(0.05)
    with profiler.stage("quick"):
        pass
    table = profiler.table()
    assert list(table.index) == ["sleepy", "quick"] and list(table.columns) == ["seconds"]
    assert table.loc["sleepy", "seconds"] >= 0.04 and profiler.total_seconds() >= 0.04


def test_profiler_records_a_stage_even_when_it_raises():
    profiler = StageProfiler()
    with pytest.raises(RuntimeError), profiler.stage("boom"):
        raise RuntimeError("fail")
    assert profiler.records[0]["stage"] == "boom"


def test_profiler_tracks_peak_memory_when_asked():
    profiler = StageProfiler(track_memory=True)
    with profiler.stage("allocate"):
        block = np.ones(5_000_000)  # about 40 MB
        del block
    with profiler.stage("nothing"):
        pass
    table = profiler.table()
    assert table.loc["allocate", "peak_memory_mb"] > 30
    assert table.loc["nothing", "peak_memory_mb"] < 5


# --------------------------------------------------------------------------- #
# Run record
# --------------------------------------------------------------------------- #
def test_run_recorder_writes_a_complete_record(config, tmp_path):
    data_path = tmp_path / "d.csv"
    data = pd.DataFrame({"a": [1, 2, 3]})
    data.to_csv(data_path, index=False)
    profiler = StageProfiler()
    with profiler.stage("work"):
        pass
    rec = RunRecorder("rid-1", config, ["--features", "--seed", "1"])
    rec.set_data(data_path, data)
    rec.add(
        "comparison",
        {"table": pd.DataFrame({"rmse": [1.0, np.nan]}, index=["a", "b"]), "fold_rmse": {"a": np.array([1.0, 2.0])}},
    )
    path = rec.write(tmp_path / "reports" / "rid-1", profiler)
    record = json.loads(path.read_text())
    assert record["run_id"] == "rid-1" and record["command"] == ["main.py", "--features", "--seed", "1"]
    assert record["config"]["selection"]["method"] == "vif+rfe" and record["config_hash"] == config_hash(config)
    assert record["data"]["rows"] == 3 and record["data"]["sha256"] == file_sha256(data_path)
    assert record["timings"]["stages"][0]["stage"] == "work"
    assert record["comparison"]["fold_rmse"]["a"] == [1.0, 2.0] and record["comparison"]["table"]["b"]["rmse"] is None
    assert {"python", "platform", "packages"} <= set(record["environment"])
    assert record["started_utc"] <= record["finished_utc"]


def test_run_registry_appends_and_handles_new_columns(config, tmp_path):
    base = tmp_path / "reports"
    RunRecorder("run-a", config, ["--features"]).register(base, {"rows": 1000, "selection": "vif"})
    path = RunRecorder("run-b", config, ["--validate"]).register(base, {"rows": 1000, "validation_passed": True})
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["run_id"] for r in rows] == ["run-a", "run-b"]
    assert rows[0]["selection"] == "vif" and rows[0]["validation_passed"] == ""  # columns are unioned
    assert rows[1]["validation_passed"] == "True" and rows[1]["config_hash"] == config_hash(config)


def test_ensure_unique_run_id_avoids_existing_folders(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert ensure_unique_run_id("run", [tmp_path / "a", tmp_path / "b"]) == "run"
    (tmp_path / "a" / "run").mkdir()
    assert ensure_unique_run_id("run", [tmp_path / "a", tmp_path / "b"]) == "run-2"
    (tmp_path / "b" / "run-2").mkdir()  # a collision in a different base folder counts too
    assert ensure_unique_run_id("run", [tmp_path / "a", tmp_path / "b"]) == "run-3"
