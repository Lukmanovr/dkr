# DKR course build tooling.
# Targets work on Windows (Git Bash / make from Git for Windows) and Linux CI alike.

PY ?= python

.PHONY: site preview test linkcheck clean

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

clean:
	rm -rf _site .quarto
