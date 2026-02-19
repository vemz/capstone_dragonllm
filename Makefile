PYTHON = python3
VENV_DIR = .venv
VENV_PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
VLLM = $(VENV_DIR)/bin/vllm

GENERATOR_MODEL = Qwen/Qwen3-30B-A3B-Base
GUARD_MODEL = Qwen/Qwen3Guard-Gen-8B
PORT = 8082

SYNTHESIS_SCRIPT = reproduction_model/synthesis.py
LABELLING_SCRIPT = reproduction_model/labelling.py

.PHONY: venv install serve-generator serve-guard synthesis labelling pipeline clean kill-server

venv:
	$(PYTHON) -m venv $(VENV_DIR)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install vllm openai datasets transformers tqdm

synthesis:
	$(VLLM) serve $(GENERATOR_MODEL) --port $(PORT) &
	$(VENV_PYTHON) $(SYNTHESIS_SCRIPT)

labelling:
	$(VLLM) serve $(GUARD_MODEL) --port $(PORT) &
	$(VENV_PYTHON) $(LABELLING_SCRIPT)

kill-server:
	@pkill -f "vllm serve" || true

clean:
	rm -rf reproduction_model/results/synthesis
	rm -rf reproduction_model/results/labelling
