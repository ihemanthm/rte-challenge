# Single entry point. Everything runs inside .venv; the system Python is never used.
UV      ?= uv
VENV    := .venv
PY      := $(VENV)/bin/python
ZIP     ?= ../Dataset for Challenge-20261101.zip
CONFIG  ?= configs/student_deberta_v3_xsmall.yaml
RUN     ?=

.PHONY: help venv lock data baselines train distill predict analyse prune submission figures test lint clean

help:
	@echo "make venv | lock | data | baselines | train CONFIG=... | distill CONFIG=... | predict RUN=... | analyse RUN=... | test | lint"

$(VENV)/bin/python:
	$(UV) venv --python 3.12 $(VENV)

venv: $(VENV)/bin/python requirements.txt requirements-dev.txt
	$(UV) pip install --python $(PY) -r requirements.txt -r requirements-dev.txt
	$(UV) pip install --python $(PY) --no-deps -e .

# Re-resolve pins after editing requirements*.in
lock:
	$(UV) pip compile --universal --python-version 3.12 requirements.in -o requirements.txt
	$(UV) pip compile --universal --python-version 3.12 requirements-dev.in -o requirements-dev.txt

requirements.txt: requirements.in
	$(UV) pip compile --universal --python-version 3.12 requirements.in -o requirements.txt

requirements-dev.txt: requirements-dev.in requirements.txt
	$(UV) pip compile --universal --python-version 3.12 requirements-dev.in -o requirements-dev.txt

data:
	$(PY) scripts/prepare_data.py --zip "$(ZIP)" --out data

baselines:
	$(PY) -m rte.baselines --train data/train.jsonl --dev data/dev.jsonl

train:
	$(PY) -m rte.train --config $(CONFIG)

distill:
	$(PY) -m rte.distill --config $(CONFIG)

predict:
	$(PY) -m rte.predict --checkpoint $(RUN) --input data/dev.jsonl

analyse:
	$(PY) -m rte.analysis --predictions $(RUN)/dev_predictions.jsonl

# Submission = plain xsmall student (checkpoint chosen by validation) with a pruned vocabulary
SUB_RUN ?= outputs/student_deberta_v3_xsmall

prune:
	$(PY) -m rte.prune --checkpoint $(SUB_RUN) --out $(SUB_RUN)_pruned

submission: prune
	$(PY) scripts/export_predictions.py --run $(SUB_RUN)_pruned

figures:
	$(PY) scripts/make_figures.py --pruned $(SUB_RUN)_pruned --out report

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src scripts tests

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache
