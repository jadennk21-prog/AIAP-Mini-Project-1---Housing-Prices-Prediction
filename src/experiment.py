"""
Experiment utilities: configuration files, run IDs, profiling and machine-readable run records.

* **Config files** (YAML or JSON) describe a whole experiment, so it can be rerun exactly and
  version-controlled. Every run also writes the *resolved* configuration next to its results.
* **Run IDs** (``20261105-142530-3fa9c1d2``: UTC timestamp plus a hash of the configuration) keep the
  outputs of different runs from overwriting each other.
* **Profiling** records wall-clock time (and optionally peak Python memory) for each stage.
* **Run records** (``results.json``) bundle metadata, configuration, data fingerprint, timings and
  results (including fold-level scores) in one file.

Scope note: this project deliberately has no temporal (date/time) features. The dataset contains no
timestamp columns, so lag, seasonality or age-since features would have nothing to build on.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import platform
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "CONFIG_SECTIONS",
    "load_config",
    "save_config",
    "config_hash",
    "file_sha256",
    "make_run_id",
    "ensure_unique_run_id",
    "versioned_dir",
    "to_jsonable",
    "environment_info",
    "StageProfiler",
    "RunRecorder",
]

logger = logging.getLogger(__name__)

#: Top-level sections accepted in a config file.
CONFIG_SECTIONS = ("run", "preprocessing", "feature_engineering", "selection", "cv")
_RUN_KEYS = ("data",)
_SELECTION_KEYS = ("method",)
_CV_KEYS = ("folds", "repeats")


# --------------------------------------------------------------------------- #
# Config files
# --------------------------------------------------------------------------- #
def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML (``.yaml`` / ``.yml``) or JSON (``.json``) experiment configuration.

    Unknown sections or keys raise ``ValueError`` so typos are not silently ignored. The
    ``preprocessing`` and ``feature_engineering`` sections are validated further when they are
    turned into config objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Reading YAML configs needs PyYAML: pip install pyyaml (or use a .json config)") from exc
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format '{path.suffix}'. Use .yaml, .yml or .json")
    data = {} if data is None else data
    if not isinstance(data, dict):
        raise ValueError("A config file must contain a mapping of sections")

    unknown = sorted(set(data) - set(CONFIG_SECTIONS))
    if unknown:
        raise ValueError(f"Unknown config section(s) {unknown}; valid sections: {list(CONFIG_SECTIONS)}")
    for section in CONFIG_SECTIONS:
        if section in data and data[section] is not None and not isinstance(data[section], dict):
            raise ValueError(f"Config section '{section}' must be a mapping")
    for section, allowed in (("run", _RUN_KEYS), ("selection", _SELECTION_KEYS), ("cv", _CV_KEYS)):
        bad = sorted(set(data.get(section) or {}) - set(allowed))
        if bad:
            raise ValueError(f"Unknown key(s) {bad} in config section '{section}'; valid keys: {list(allowed)}")
    return data


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Write a configuration as YAML (``.yaml`` / ``.yml``) or JSON, creating folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = to_jsonable(config)
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        path.write_text(yaml.safe_dump(safe, sort_keys=False, default_flow_style=False), encoding="utf-8")
    elif path.suffix.lower() == ".json":
        path.write_text(json.dumps(safe, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported config format '{path.suffix}'. Use .yaml, .yml or .json")
    return path


def config_hash(config: Dict[str, Any], length: int = 8) -> str:
    """Short, stable fingerprint of a configuration (independent of key order)."""
    canonical = json.dumps(to_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: Union[str, Path]) -> str:
    """SHA-256 of a file, used to record exactly which data a run used."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Run IDs and versioned folders
# --------------------------------------------------------------------------- #
def make_run_id(config: Dict[str, Any], now: Optional[datetime] = None, label: Optional[str] = None) -> str:
    """``YYYYmmdd-HHMMSS-<config hash>`` (UTC), optionally followed by ``-<label>``."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{config_hash(config)}"
    if label:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in label)
        run_id = f"{run_id}-{safe}"
    return run_id


def ensure_unique_run_id(run_id: str, bases: Sequence[Union[str, Path]]) -> str:
    """Append ``-2``, ``-3`` ... if a folder named ``run_id`` already exists in any of ``bases``.

    Two runs with the same configuration in the same second would otherwise share an ID and
    overwrite each other's outputs.
    """
    candidate, counter = run_id, 1
    while any((Path(base) / candidate).exists() for base in bases):
        counter += 1
        candidate = f"{run_id}-{counter}"
    return candidate


def versioned_dir(base: Union[str, Path], run_id: str, enabled: bool = True) -> Path:
    """Create and return ``base/run_id`` (versioning on) or ``base`` (off).

    With versioning on, ``base/latest.txt`` is kept pointing at the newest run.
    """
    base = Path(base)
    if not enabled:
        base.mkdir(parents=True, exist_ok=True)
        return base
    target = base / run_id
    target.mkdir(parents=True, exist_ok=True)
    (base / "latest.txt").write_text(run_id + "\n", encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# JSON helpers and environment
# --------------------------------------------------------------------------- #
def to_jsonable(obj: Any) -> Any:
    """Convert numpy / pandas / Path / dataclass-like objects (and NaN) into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return {str(idx): to_jsonable(row) for idx, row in obj.to_dict(orient="index").items()}
    if isinstance(obj, pd.Series):
        return {str(k): to_jsonable(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if obj is None or isinstance(obj, (int, str)):
        return obj
    return str(obj)


def environment_info() -> Dict[str, Any]:
    """Python, platform and key package versions, so a run can be reproduced later."""
    packages: Dict[str, Optional[str]] = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "statsmodels", "joblib", "PyYAML"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": packages}


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #
@dataclass
class StageProfiler:
    """Record wall-clock time (and optionally peak Python memory) of named stages.

    ``track_memory=True`` uses :mod:`tracemalloc`, which sees Python and NumPy allocations but adds
    runtime overhead, so it is off by default::

        profiler = StageProfiler()
        with profiler.stage("preprocessing"):
            ...
        profiler.table()
    """

    track_memory: bool = False
    records: List[Dict[str, Any]] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started_tracing = False
        if self.track_memory and not tracemalloc.is_tracing():
            tracemalloc.start()
            started_tracing = True
        if self.track_memory:
            tracemalloc.reset_peak()
        start = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - start
            peak_mb: Optional[float] = None
            if self.track_memory:
                peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
                if started_tracing:
                    tracemalloc.stop()
            self.records.append(
                {
                    "stage": name,
                    "seconds": round(seconds, 3),
                    "peak_memory_mb": None if peak_mb is None else round(peak_mb, 1),
                }
            )
            logger.info(
                "Stage '%s' took %.2fs%s",
                name,
                seconds,
                "" if peak_mb is None else f" (peak Python memory {peak_mb:.0f} MB)",
            )

    def table(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.records, columns=["stage", "seconds", "peak_memory_mb"])
        if not self.track_memory:
            frame = frame.drop(columns="peak_memory_mb")
        return frame.set_index("stage")

    def total_seconds(self) -> float:
        return round(sum(r["seconds"] for r in self.records), 3)


# --------------------------------------------------------------------------- #
# Run record
# --------------------------------------------------------------------------- #
class RunRecorder:
    """Collect metadata and results of one run and write them as ``results.json``.

    The record contains the run ID, UTC start time, command line, environment, resolved
    configuration, a fingerprint of the data, stage timings and every result added with
    :meth:`add` (for example cross-validation tables including fold-level scores).
    """

    def __init__(self, run_id: str, config: Dict[str, Any], argv: Optional[Sequence[str]] = None):
        self.run_id = run_id
        self.config = config
        self.argv = list(argv) if argv is not None else []
        self.started = datetime.now(timezone.utc)
        self.data: Dict[str, Any] = {}
        self.sections: Dict[str, Any] = {}

    def set_data(self, path: Union[str, Path], df: pd.DataFrame) -> None:
        self.data = {
            "path": str(path),
            "sha256": file_sha256(path),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        }

    def add(self, name: str, payload: Any) -> None:
        self.sections[name] = to_jsonable(payload)

    def as_dict(self, profiler: Optional[StageProfiler] = None) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "run_id": self.run_id,
            "started_utc": self.started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "command": ["main.py", *self.argv],
            "environment": environment_info(),
            "config": to_jsonable(self.config),
            "config_hash": config_hash(self.config),
            "data": self.data,
        }
        if profiler is not None:
            record["timings"] = {"stages": profiler.records, "total_seconds": profiler.total_seconds()}
        record.update(self.sections)
        return to_jsonable(record)

    def write(self, directory: Union[str, Path], profiler: Optional[StageProfiler] = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "results.json"
        path.write_text(json.dumps(self.as_dict(profiler), indent=2), encoding="utf-8")
        return path

    def register(self, reports_base: Union[str, Path], summary: Optional[Dict[str, Any]] = None) -> Path:
        """Append one line to ``<reports_base>/runs.csv`` so all runs can be listed and compared."""
        reports_base = Path(reports_base)
        reports_base.mkdir(parents=True, exist_ok=True)
        path = reports_base / "runs.csv"
        row = {
            "run_id": self.run_id,
            "started_utc": self.started.isoformat(),
            "config_hash": config_hash(self.config),
            "command": " ".join(self.argv),
            **(summary or {}),
        }
        row = {k: to_jsonable(v) for k, v in row.items()}
        existing: List[Dict[str, Any]] = []
        if path.exists():
            with open(path, newline="", encoding="utf-8") as handle:
                existing = list(csv.DictReader(handle))
        rows = existing + [row]
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path
