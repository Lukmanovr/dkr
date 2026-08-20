#!/usr/bin/env python
"""Guard: no solution content may ever land in the public repo.

Solution notebooks live only in the private repo and carry these markers.
CI runs this on every push/PR of the public repo and fails if any marker
(or a solution-named file) appears.

Usage: python scripts/check_no_solutions.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MARKERS = (
    "### BEGIN SOLUTION",
    "### END SOLUTION",
    "# SOLUTION:",
    "<!-- solution -->",
)
FILENAME_PATTERN = re.compile(r"solution", re.IGNORECASE)
# Public sample exams ship WITH solutions by design; anything named sample_* is exempt.
EXEMPT_PATTERN = re.compile(r"sample", re.IGNORECASE)
SCAN_DIRS = ("labs", "homeworks", "lectures", "exams", "project")
SELF = Path(__file__).resolve()


def main() -> int:
    root = SELF.parent.parent
    offenders: list[str] = []

    for d in SCAN_DIRS:
        base = root / d
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if EXEMPT_PATTERN.search(path.name):
                continue
            if FILENAME_PATTERN.search(path.name):
                offenders.append(f"{path.relative_to(root)} (solution-named file)")
                continue
            if path.suffix.lower() in {".ipynb", ".qmd", ".md", ".py", ".tex", ".html"}:
                text = path.read_text(encoding="utf-8", errors="replace")
                for marker in MARKERS:
                    if marker in text:
                        offenders.append(f"{path.relative_to(root)} (contains '{marker}')")
                        break

    if offenders:
        print("Solution content detected in the PUBLIC repo — move it to dkr-private:")
        for o in offenders:
            print(f"  - {o}")
        return 1
    print("No solution content in public tree — ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
