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

serve-generator:
	$(VLLM) serve $(GENERATOR_MODEL) --port $(PORT) &

serve-guard:
	$(VLLM) serve $(GUARD_MODEL) --port $(PORT) &

synthesis:
	$(VENV_PYTHON) $(SYNTHESIS_SCRIPT)

labelling:
	$(VENV_PYTHON) $(LABELLING_SCRIPT)

kill-server:
	@pkill -f "vllm serve" || true

pipeline:
	@echo "=== Step 1: Starting generator server ==="
	$(MAKE) serve-generator
	@echo "Waiting for server to be ready..."
	@sleep 30
	@echo "=== Step 2: Running synthesis ==="
	$(MAKE) synthesis
	@echo "=== Step 3: Stopping generator server ==="
	$(MAKE) kill-server
	@sleep 5
	@echo "=== Step 4: Starting guard server ==="
	$(MAKE) serve-guard
	@echo "Waiting for server to be ready..."
	@sleep 30
	@echo "=== Step 5: Running labelling ==="
	$(MAKE) labelling
	@echo "=== Step 6: Stopping guard server ==="
	$(MAKE) kill-server
	@echo "=== Pipeline complete ==="

clean:
	rm -f reproduction_model/rtp_synthesized_mixed_4B.jsonl
	rm -f reproduction_model/rtp_labeled_mixed.jsonl
