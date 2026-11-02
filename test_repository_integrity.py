"""The repository must be complete and its documentation must match the code.

These tests exist because a submission once lacked modules that ``main.py`` imports, and because documentation
that drifts from the code erodes trust. They check completeness (files, imports, requirements) and compare the
README with the real files, command-line options and test counts.
"""

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import main

ROOT = Path(main.ROOT)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "verify_clean_checkout", ROOT / "scripts" / "verify_clean_checkout.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_clean_checkout"] = module
    spec.loader.exec_module(module)
    return module


verify = _load_script()
README = (ROOT / "README.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Completeness of the real repository
# --------------------------------------------------------------------------- #
def test_all_required_files_exist():
    assert verify.find_missing_files(ROOT) == []


def test_every_internal_import_resolves_to_a_real_file_and_name():
    assert verify.find_import_problems(ROOT) == []


def test_requirements_list_every_third_party_import():
    assert verify.find_unlisted_requirements(ROOT) == []


def test_main_imports_only_modules_that_exist():
    for module in ("experiment", "feature_engineering", "preprocessing", "models", "evaluation"):
        assert (ROOT / "src" / f"{module}.py").exists(), module


# --------------------------------------------------------------------------- #
# The checks really detect problems (on tampered copies)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def tampered_copy(tmp_path):
    copy = tmp_path / "copy"
    verify.copy_repository(ROOT, copy)
    return copy


def test_a_missing_module_is_detected(tampered_copy):
    (tampered_copy / "src" / "experiment.py").unlink()
    problems = verify.static_problems(tampered_copy)
    assert any("src/experiment.py" in p and "missing" in p for p in problems)
    assert any("main.py" in p and "src.experiment" in p for p in problems)


def test_a_missing_function_is_detected(tampered_copy):
    path = tampered_copy / "src" / "models.py"
    path.write_text(path.read_text().replace("def run_benchmark(", "def run_benchmark_renamed("))
    problems = verify.find_import_problems(tampered_copy)
    assert any("'run_benchmark' is not defined in src/models.py" in p for p in problems)


def test_a_missing_requirement_is_detected(tampered_copy):
    path = tampered_copy / "requirements.txt"
    path.write_text("\n".join(line for line in path.read_text().splitlines() if "scikit-learn" not in line))
    assert any("sklearn" in p for p in verify.find_unlisted_requirements(tampered_copy))


def test_a_missing_required_file_is_detected(tampered_copy):
    (tampered_copy / "src" / "preprocessing.py").unlink()
    (tampered_copy / "Housing_Price_Dataset.csv").unlink()
    missing = verify.find_missing_files(tampered_copy)
    assert "src/preprocessing.py" in missing and "Housing_Price_Dataset.csv" in missing


def test_copy_repository_leaves_out_generated_files(tmp_path):
    source = tmp_path / "src_repo"
    (source / "reports" / "run1").mkdir(parents=True)
    (source / "reports" / "run1" / "results.json").write_text("{}")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "x.pyc").write_bytes(b"0")
    (source / "keep.py").write_text("x = 1")
    copied = verify.copy_repository(source, tmp_path / "dest")
    assert [str(p) for p in copied] == ["keep.py"]


def test_the_script_reports_failure_for_an_incomplete_repository(tampered_copy, capsys):
    shutil.rmtree(tampered_copy / "src")
    code = verify.main(["--root", str(tampered_copy), "--quick"])
    assert code == 1 and "incomplete" in capsys.readouterr().out


def test_the_script_passes_quickly_on_the_real_repository(capsys):
    assert verify.main(["--quick"]) == 0
    assert "PASS" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Documentation matches the code
# --------------------------------------------------------------------------- #
def _tree_paths(readme: str):
    """Paths listed in the README's 'Repository structure' tree."""
    block = re.search(r"## Repository structure\s+```\n(.*?)```", readme, re.S).group(1)
    stack, paths = [], []
    for line in block.splitlines()[1:]:
        match = re.match(r"^((?:│   |    )*)(?:├── |└── )(\S+)", line)
        if not match:
            continue
        depth = len(match.group(1)) // 4
        del stack[depth:]
        stack.append(match.group(2).rstrip("/"))
        paths.append("/".join(stack))
    return paths


def test_every_file_in_the_readme_tree_exists():
    paths = _tree_paths(README)
    assert len(paths) > 20
    missing = [p for p in paths if not (ROOT / p).exists()]
    assert missing == []


def test_every_python_module_is_documented_in_the_readme_tree():
    listed = set(_tree_paths(README))
    actual = {
        str(p.relative_to(ROOT))
        for folder in ("src", "tests", "scripts", "examples")
        for p in (ROOT / folder).glob("*.py")
    }
    undocumented = sorted(p for p in actual - listed if not p.endswith("__init__.py"))
    assert undocumented == []


def test_every_command_line_option_in_the_readme_exists():
    options = set(main.build_parser()._option_string_actions)
    reference = re.search(r"### Command-line reference(.*?)###", README, re.S).group(1)
    quick = re.search(r"## Quick start\s+```bash\n(.*?)```", README, re.S).group(1)
    mentioned = set(re.findall(r"`(--[a-z][a-z-]*)", reference)) | {
        flag for line in quick.splitlines() if "main.py" in line for flag in re.findall(r"(--[a-z][a-z-]*)", line)
    }
    assert len(mentioned) > 15
    assert sorted(mentioned - options) == []


def test_every_cli_option_is_documented_in_the_readme():
    documented = set(re.findall(r"(--[a-z][a-z-]*)", README))
    options = {o for o in main.build_parser()._option_string_actions if o.startswith("--") and o != "--help"}
    assert sorted(options - documented) == []


def test_readme_test_counts_match_the_real_test_suite():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    counts = {}
    for line in result.stdout.splitlines():
        if "::" in line:
            counts[Path(line.split("::")[0]).name] = counts.get(Path(line.split("::")[0]).name, 0) + 1
    stated = {name: int(n) for name, n in re.findall(r"(test_\w+\.py)\s+# (\d+) tests", README)}
    assert set(stated) == set(counts), "every test file should have a count in the README tree"
    assert stated == {name: counts[name] for name in stated}
    total = int(re.search(r"# run the (\d+) unit tests", README).group(1))
    assert total == sum(counts.values())


def test_readme_states_the_problem_is_regression():
    head = README[: README.index("## Quick start")]
    assert "regression" in head.lower() and "not applicable" in head.lower()


def test_config_files_in_the_readme_tree_load():
    from src.experiment import load_config

    configs = [p for p in _tree_paths(README) if p.startswith("configs/") and p.endswith(".yaml")]
    assert configs
    for path in configs:
        load_config(ROOT / path)
