"""Verify that the repository is complete by running it from a clean copy.

Run from the repository root before submitting::

    python scripts/verify_clean_checkout.py              # check completeness, run the tests and the validation suite
    python scripts/verify_clean_checkout.py --zip submission.zip   # ... and write a zip of exactly what was verified
    (add --zip-root NAME to choose the folder name inside the zip)
    python scripts/verify_clean_checkout.py --quick      # only the static checks (no tests)

It does three things:

1. **Static completeness checks**: every required file exists; every ``from src.x import name`` /
   ``from .x import name`` in ``main.py``, ``src/`` and ``tests/`` points at a file and a name that really exists;
   every third-party package that is imported is listed in ``requirements.txt``.
2. **Copy the repository to a temporary folder** (without generated outputs, caches or ``.git``), so nothing that
   only exists on your machine can hide a missing file.
3. **Run the unit tests and ``main.py --validate`` inside that copy.**

A missing module (for example ``src/experiment.py`` not being in a submission) fails step 1 with a clear message.
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]

#: Files the project needs in order to run and be assessed.
REQUIRED_FILES = [
    "README.md",
    "eda.ipynb",
    "requirements.txt",
    "main.py",
    "Housing_Price_Dataset.csv",
    "src/__init__.py",
    "src/preprocessing.py",
    "src/feature_engineering.py",
    "src/models.py",
    "src/evaluation.py",
    "src/experiment.py",
    "tests/conftest.py",
]

#: Folders and files that are generated or machine-specific and are not part of a submission.
EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
    "processed",
    "artifacts",
    "reports",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}

#: import name -> requirement name, where they differ
IMPORT_TO_REQUIREMENT = {"sklearn": "scikit-learn", "yaml": "pyyaml", "PIL": "pillow"}


def _py_files(root: Path) -> List[Path]:
    files = [root / "main.py"] if (root / "main.py").exists() else []
    for folder in ("src", "tests", "scripts", "examples"):
        files += sorted((root / folder).rglob("*.py")) if (root / folder).exists() else []
    return files


def _top_level_names(path: Path) -> Set[str]:
    """Names a module defines at its top level (functions, classes, assignments, imports, ``__all__``)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        names.add(sub.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def find_missing_files(root: Path = ROOT) -> List[str]:
    """Required files that are not present."""
    return [name for name in REQUIRED_FILES if not (root / name).exists()]


def find_import_problems(root: Path = ROOT) -> List[str]:
    """Internal imports (``src.x`` / ``.x``) that point at a missing file or a name the module does not define."""
    problems: List[str] = []
    for path in _py_files(root):
        rel = path.relative_to(root)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module: Optional[str] = None
            if isinstance(node, ast.ImportFrom):
                if node.level == 1 and node.module and rel.parts[0] == "src":
                    module = f"src.{node.module}"
                elif node.level == 0 and node.module and node.module.split(".")[0] == "src":
                    module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "src":
                        target = root / (alias.name.replace(".", "/") + ".py")
                        if not target.exists() and not (root / alias.name.replace(".", "/") / "__init__.py").exists():
                            problems.append(f"{rel}:{node.lineno}: 'import {alias.name}' but that module is missing")
                continue
            if module is None:
                continue
            target = root / (module.replace(".", "/") + ".py")
            if not target.exists():
                if (root / module.replace(".", "/") / "__init__.py").exists():
                    continue
                problems.append(
                    f"{rel}:{node.lineno}: imports from '{module}' but {target.relative_to(root)} is missing"
                )
                continue
            defined = _top_level_names(target)
            assert isinstance(node, ast.ImportFrom)
            for alias in node.names:
                if alias.name != "*" and alias.name not in defined:
                    problems.append(f"{rel}:{node.lineno}: '{alias.name}' is not defined in {target.relative_to(root)}")
    return problems


def _requirement_names(root: Path) -> Set[str]:
    path = root / "requirements.txt"
    if not path.exists():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            names.add(re.split(r"[<>=!~\[; ]", line)[0].lower().replace("_", "-"))
    return names


def find_unlisted_requirements(root: Path = ROOT) -> List[str]:
    """Third-party packages imported by the code but not listed in ``requirements.txt``."""
    listed = _requirement_names(root)
    stdlib = set(sys.stdlib_module_names)
    local = {"src", "main", "tests", "scripts", "examples"} | {p.stem for p in (root / "scripts").glob("*.py")}
    wanted: Dict[str, Path] = {}
    for path in _py_files(root):
        if path.relative_to(root).parts[0] in ("scripts", "examples"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: List[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name not in stdlib and name not in local and name != "__future__":
                    wanted.setdefault(name, path)
    missing = []
    for name, first_use in sorted(wanted.items()):
        requirement = IMPORT_TO_REQUIREMENT.get(name, name).lower().replace("_", "-")
        if requirement not in listed:
            missing.append(
                f"'{name}' is imported in {first_use.relative_to(root)} but '{requirement}' is not in requirements.txt"
            )
    return missing


def static_problems(root: Path = ROOT) -> List[str]:
    return (
        [f"missing required file: {name}" for name in find_missing_files(root)]
        + find_import_problems(root)
        + find_unlisted_requirements(root)
    )


def copy_repository(root: Path, destination: Path) -> List[Path]:
    """Copy the repository without generated files; returns the copied files."""
    copied = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.is_dir() or any(part in EXCLUDE_DIRS for part in rel.parts) or path.suffix in EXCLUDE_SUFFIXES:
            continue
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(rel)
    return copied


def _run(command: List[str], cwd: Path) -> bool:
    env = {"PYTHONDONTWRITEBYTECODE": "1", "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin"}
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-4:])
    print(f"    {tail}")
    return result.returncode == 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, help="write a zip of the verified files to this path")
    parser.add_argument(
        "--zip-root", help="name of the top-level folder inside the zip (default: the repository folder name)"
    )
    parser.add_argument("--quick", action="store_true", help="only the static checks (no tests)")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.root.resolve()

    print(f"Checking {root}")
    problems = static_problems(root)
    if problems:
        print("\nFAIL: the repository is incomplete or inconsistent:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("  [PASS] all required files exist, every internal import resolves, requirements.txt covers the imports")
    if args.quick:
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "clean_checkout"
        files = copy_repository(root, copy)
        print(f"  [PASS] copied {len(files)} files to a clean folder")
        print("  Running the unit tests in the clean copy ...")
        if not _run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], copy):
            print("FAIL: unit tests failed in the clean copy")
            return 1
        print("  Running main.py --validate in the clean copy ...")
        if not _run([sys.executable, "main.py", "--validate", "--no-save", "--quiet"], copy):
            print("FAIL: the validation suite failed in the clean copy")
            return 1
        if args.zip:
            args.zip.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as archive:
                for rel in files:
                    archive.write(copy / rel, Path(args.zip_root or root.name) / rel)
            print(f"  [PASS] wrote {args.zip} ({len(files)} files)")
    print("\nPASS: the repository runs from a clean copy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
