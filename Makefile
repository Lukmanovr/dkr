# DKR course build tooling.
# Targets work on Windows (Git Bash / make from Git for Windows) and Linux CI alike.

PY ?= python

.PHONY: site preview test linkcheck qa clean

site:
	quarto render

preview:
	quarto preview

# Execute every lab/homework notebook headlessly. SMOKE=1 makes notebooks use
# reduced epochs/subsets (each notebook reads the SMOKE env var in its config cell).
test:
	SMOKE=1 $(PY) scripts/run_notebooks.py labs homeworks

linkcheck:
	$(PY) scripts/check_links.py _site

# The full pre-commit battery (everything CI runs except the notebooks).
# Run `make site` first — the link check reads _site/.
qa:
	$(PY) scripts/check_no_solutions.py
	$(PY) scripts/check_links.py _site --skip-external
	node scripts/figlint.mjs
	node scripts/widget_smoke.mjs

clean:
	rm -rf _site .quarto
