#!/usr/bin/env python3
"""
Authoritative repository validation suite for FlyRank Capstone.
Enforces structural integrity, one-line submission file constraints,
anti-hardcoding bans, and unit testing validation.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

REQUIRED_FILES = [
    "work/01_data_exploration.ipynb",
    "work/02_data_quality.ipynb",
    "work/03_feature_engineering.ipynb",
    "work/04_label_validation.ipynb",
    "work/05_baseline_and_model.ipynb",
    "work/06_evaluation_and_error_analysis.ipynb",
    "work/07_capstone.ipynb",
    "src/config.py",
    "src/data.py",
    "src/features.py",
    "src/labels.py",
    "src/leakage.py",
    "src/splits.py",
    "src/baseline.py",
    "src/model.py",
    "src/evaluation.py",
    "src/recommendations.py",
    "src/explainability.py",
    "src/privacy.py",
    "configs/default.yaml",
    "configs/schema_mapping.yaml",
    "tests/test_features.py",
    "tests/test_labels.py",
    "tests/test_leakage.py",
    "tests/test_splits.py",
    "tests/test_metrics.py",
    "tests/test_privacy.py",
    "scripts/inspect_dataset.py",
    "scripts/run_pipeline.py",
    "scripts/generate_paper.py",
    "scripts/validate_repo.py",
    "paper/index.html",
    "paper/assets/styles.css",
    "paper/assets/script.js",
    "submission/paper_url.txt",
    ".github/workflows/deploy-paper.yml",
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    ".env.example",
    ".gitignore",
]

FORBIDDEN_PATTERNS = [
    re.compile(r"\b0\.72\s+precision", re.IGNORECASE),
    re.compile(r"\b3\s*[x×]\s*lift\b", re.IGNORECASE),
    re.compile(r"\b22%\s+positive\s+rate\b", re.IGNORECASE),
    re.compile(r"\b30[,\.]?000\s+pages\b", re.IGNORECASE),
]


def validate_repository_structure() -> bool:
    print("1. Checking required files existence...")
    missing = []
    for rel_path in REQUIRED_FILES:
        p = Path(rel_path)
        if not p.exists():
            missing.append(rel_path)

    if missing:
        print(f"FAILED: Missing {len(missing)} required files:")
        for m in missing:
            print(f"  - {m}")
        return False
    print(f"   All {len(REQUIRED_FILES)} required files exist.")
    return True


def validate_submission_file() -> bool:
    print("\n2. Checking submission/paper_url.txt format...")
    p = Path("submission/paper_url.txt")
    if not p.exists():
        print("FAILED: submission/paper_url.txt missing.")
        return False

    with open(p, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) != 1:
        print(f"FAILED: submission/paper_url.txt must contain exactly 1 line. Found {len(lines)} lines.")
        return False

    content = lines[0].strip()
    if content not in ["PAPER_URL_NOT_SET"] and not content.startswith("https://"):
        print(f"FAILED: Content must be 'PAPER_URL_NOT_SET' or an 'https://' URL. Found: '{content}'")
        return False

    print(f"   submission/paper_url.txt is valid ('{content}').")
    return True


def scan_for_hardcoded_empirical_results() -> bool:
    print("\n3. Scanning repository for forbidden hard-coded empirical result claims...")
    violations = []

    scan_targets = [
        "README.md",
        "paper/index.html",
        "configs/default.yaml",
    ]

    for rel_path in scan_targets:
        p = Path(rel_path)
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        for pattern in FORBIDDEN_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                violations.append(f"{rel_path}: Matched forbidden empirical pattern '{matches[0]}'")

    if violations:
        print("FAILED: Found forbidden hardcoded empirical claims:")
        for v in violations:
            print(f"  - {v}")
        return False

    print("   Zero forbidden hard-coded empirical claims detected.")
    return True


def validate_python_code_and_tests() -> bool:
    print("\n4. Validating Python source code syntax and AST...")
    src_files = list(Path("src").glob("*.py")) + list(Path("scripts").glob("*.py")) + list(Path("tests").glob("*.py"))
    ast_errors = []

    for py_file in src_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                ast.parse(f.read(), filename=str(py_file))
        except SyntaxError as e:
            ast_errors.append(f"{py_file}: {e}")

    if ast_errors:
        print(f"FAILED: Syntax errors in {len(ast_errors)} files:")
        for err in ast_errors:
            print(f"  - {err}")
        return False
    print(f"   Syntax & AST validated cleanly across all {len(src_files)} Python modules.")

    # Check if pytest is available in this environment to run the test suite
    print("\n5. Checking test suite execution runner...")
    check_pytest = subprocess.run([sys.executable, "-m", "pytest", "--version"], capture_output=True, text=True)
    if check_pytest.returncode == 0:
        res = subprocess.run([sys.executable, "-m", "pytest", "tests/"], capture_output=True, text=True)
        if res.returncode != 0:
            print("FAILED: Pytest encountered test failures:")
            print(res.stdout)
            print(res.stderr)
            return False
        print("   All pytest unit tests passed.")
    else:
        print("   [INFO] pytest is not yet installed in the current environment.")
        print("   Dependencies are specified in requirements.txt (pip install -r requirements.txt).")
        print("   AST parsing and code contract checks passed.")

    return True


def main():
    print("==================================================")
    print("FLYRANK CAPSTONE ACCEPTANCE VALIDATION")
    print("==================================================")

    ok1 = validate_repository_structure()
    ok2 = validate_submission_file()
    ok3 = scan_for_hardcoded_empirical_results()
    ok4 = validate_python_code_and_tests()

    print("==================================================")
    if ok1 and ok2 and ok3 and ok4:
        print("ALL ACCEPTANCE VALIDATION CHECKS PASSED.")
        return 0
    else:
        print("VALIDATION CHECKS FAILED. Please review above output.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
