#!/usr/bin/env python
"""Run a single notebook under the course CI contract (single-file version of
run_notebooks.py — same rules: solution notebooks must be error-free; student
notebooks may error only in student-todo cells, cascade forgiven).

    python scripts/run_one_notebook.py path/to/nb.ipynb [timeout_seconds]
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ALLOWED_STUDENT_ERRORS = {"NotImplementedError", "AssertionError"}

path = Path(sys.argv[1])
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 1800
nb = nbformat.read(path, as_version=4)
is_solution = "solution" in str(path).lower()

client = NotebookClient(nb, timeout=timeout, allow_errors=True, kernel_name="python3",
                        resources={"metadata": {"path": str(path.parent)}})
client.execute()

problems, todo_hit = [], False
for idx, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    errors = [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]
    if not errors or todo_hit:
        continue
    tags = set(cell.get("metadata", {}).get("tags", []))
    for err in errors:
        ename = err.get("ename", "Error")
        if not is_solution and "student-todo" in tags and ename in ALLOWED_STUDENT_ERRORS:
            todo_hit = True
            continue
        problems.append(f"cell {idx}: {ename}: {err.get('evalue', '')[:300]}")

for cell in nb.cells:
    if cell.cell_type == "code":
        for o in cell.get("outputs", []):
            if o.get("output_type") == "stream":
                print(o.get("text", ""), end="")

if problems:
    print("FAIL:")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print(f"\n{path.name}: ok")
