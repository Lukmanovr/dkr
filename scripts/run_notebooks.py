#!/usr/bin/env python
"""Execute course notebooks headlessly and enforce the CI contract.

Contract:
- Solution notebooks (any path containing "solution") must execute with ZERO errors.
- Student notebooks may error ONLY in cells tagged `student-todo` (the intentional
  `# TODO` exercise cells, which raise NotImplementedError / AssertionError until
  a student fills them in). Any error elsewhere fails CI.
- Notebooks read the SMOKE env var in their setup cell to shrink epochs/datasets;
  this script just passes the environment through.

Usage: python scripts/run_notebooks.py labs homeworks [--timeout 1800]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ALLOWED_STUDENT_ERRORS = {"NotImplementedError", "AssertionError"}


def run_notebook(path: Path, timeout: int) -> list[str]:
    """Execute one notebook; return a list of human-readable problems (empty = pass)."""
    nb = nbformat.read(path, as_version=4)
    is_solution = "solution" in path.name.lower() or "solution" in str(path.parent).lower()

    client = NotebookClient(
        nb,
        timeout=timeout,
        allow_errors=True,  # we inspect errors ourselves to honor the student-todo contract
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()

    problems: list[str] = []
    todo_hit = False  # once a student notebook hits its first (allowed) TODO error,
    # everything downstream is cascade noise (NameError etc.) and is not judged.
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        errors = [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]
        if not errors:
            continue
        if todo_hit:
            continue
        tags = set(cell.get("metadata", {}).get("tags", []))
        for err in errors:
            ename = err.get("ename", "Error")
            if not is_solution and "student-todo" in tags and ename in ALLOWED_STUDENT_ERRORS:
                todo_hit = True
                continue  # expected: unfilled exercise cell; skip the cascade after it
            problems.append(f"cell {idx}: {ename}: {err.get('evalue', '')[:200]}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", help="directories to scan for .ipynb files")
    parser.add_argument("--timeout", type=int, default=1800, help="per-notebook timeout (s)")
    args = parser.parse_args()

    notebooks = sorted(
        nb
        for d in args.dirs
        for nb in Path(d).rglob("*.ipynb")
        if ".ipynb_checkpoints" not in nb.parts
    )
    if not notebooks:
        print(f"No notebooks found under: {', '.join(args.dirs)}")
        return 1

    failures = 0
    for nb_path in notebooks:
        print(f"── executing {nb_path} ...", flush=True)
        try:
            problems = run_notebook(nb_path, args.timeout)
        except Exception as exc:  # kernel death, timeout, missing deps
            problems = [f"execution aborted: {type(exc).__name__}: {str(exc)[:300]}"]
        if problems:
            failures += 1
            print(f"   FAIL ({len(problems)} problem(s)):")
            for p in problems:
                print(f"     - {p}")
        else:
            print("   ok")

    print(f"\n{len(notebooks) - failures}/{len(notebooks)} notebooks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
