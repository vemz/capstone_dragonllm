PYTHON = python3
VENV_DIR = .venv
VENV312_DIR = .venv312
VENV_PYTHON = $(VENV_DIR)/bin/python
VENV312_PYTHON = $(VENV312_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
PIP312 = $(VENV312_DIR)/bin/pip
VLLM = $(VENV312_DIR)/bin/vllm

GENERATOR_MODEL = Qwen/Qwen3-30B-A3B-Base
GUARD_MODEL = Qwen/Qwen3-4B-Instruct-2507
PORT = 8082

SYNTHESIS_SCRIPT = reproduction_model/synthesis.py
LABELLING_SCRIPT = reproduction_model/labelling2.py

.PHONY: venv install install-vllm serve-generator serve-guard synthesis labelling pipeline clean kill-server

venv:
	$(PYTHON) -m venv $(VENV_DIR)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install openai datasets transformers tqdm

install-vllm:
	$(PIP312) install --upgrade pip
	$(PIP312) install vllm openai datasets transformers tqdm

synthesis:
	$(VLLM) serve $(GENERATOR_MODEL) --port $(PORT) &
	$(VENV312_PYTHON) $(SYNTHESIS_SCRIPT)

labelling:
	$(VLLM) serve $(GUARD_MODEL) --port $(PORT) &
	$(VENV312_PYTHON) $(LABELLING_SCRIPT)

kill-server:
	@pkill -f "vllm serve" || true

clean:
	rm -rf reproduction_model/results/synthesis
	rm -rf reproduction_model/results/labelling
