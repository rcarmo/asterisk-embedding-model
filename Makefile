# Makefile for Asterisk Embedding Model Pipeline
#
# Based on: Semenov, A. (2024). Asterisk*: Keep it Simple. arXiv:2411.05691.
#
# Usage: make help (default)

.DEFAULT_GOAL := help
.ONESHELL:

SHELL := /bin/bash
PYTHON := uv run python
PIP := uv pip

# === Paths ===
DATA_DIR := data
BUILD_DIR := build
DIST_DIR := dist
TOKENIZER_DIR := $(DIST_DIR)/tokenizer

DATA_TSV := $(DATA_DIR)/data.tsv
TEACHER_DIR := $(BUILD_DIR)/teacher
MODEL_PT := $(BUILD_DIR)/model.pt
MODEL_ONNX := $(BUILD_DIR)/model.onnx
MODEL_SIMPLIFIED := $(BUILD_DIR)/model_simplified.onnx
MODEL_INT8 := $(BUILD_DIR)/model_int8.onnx
DIST_MODEL_INT8 := $(DIST_DIR)/model_int8.onnx

# === Training Config (override with make VAR=value) ===
EPOCHS ?= 5
BATCH_SIZE ?= 32
LR ?= 2e-4
PATIENCE ?= 3
VAL_SPLIT ?= 0.1
DISTILL_ALPHA ?= 0.5

# === Teacher Model (for knowledge distillation) ===
TEACHER_MODEL ?= sentence-transformers/all-MiniLM-L6-v2

.PHONY: all data teacher train export benchmark demo vendor clean clean-models clean-data install help

# === Main Targets ===

all: $(MODEL_INT8) ## Run full pipeline (data → teacher → train → export)
	@echo "✅ Full pipeline complete!"

data: $(DATA_TSV) ## Prepare training data from Newsroom dataset

teacher: $(TEACHER_DIR)/teacher_summaries.npy ## Precompute teacher embeddings for distillation

train: $(MODEL_PT) ## Train the Asterisk model with knowledge distillation

export: $(MODEL_INT8) ## Export to ONNX and quantize to INT8

benchmark: vendor ## Run inference latency benchmark
	@echo "⏱️  Running benchmark..."
	$(PYTHON) demo.py --benchmark 1000 --data $(DATA_TSV) --model $(DIST_MODEL_INT8)

demo: vendor ## Run similarity ranking demo
	@echo "🔍 Running similarity demo..."
	$(PYTHON) demo.py --sample-size 100 --data $(DATA_TSV) --model $(DIST_MODEL_INT8)

install: ## Install Python dependencies
	$(PIP) install -r requirements.txt

clean: ## Remove all generated files
	@echo "🧹 Cleaning generated files..."
	rm -rf $(DATA_DIR) $(BUILD_DIR) $(DIST_DIR)
	@echo "✅ Clean complete"

clean-models: ## Remove model files only
	rm -f $(MODEL_PT) $(MODEL_ONNX) $(MODEL_ONNX).data $(MODEL_SIMPLIFIED) $(MODEL_INT8) $(DIST_MODEL_INT8)

clean-data: ## Remove data files only
	rm -rf $(DATA_DIR) $(TEACHER_DIR)

help: ## Show this help
	@echo "Asterisk Embedding Model Pipeline"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Config (override with make VAR=value):"
	@echo "  EPOCHS=$(EPOCHS)  BATCH_SIZE=$(BATCH_SIZE)  LR=$(LR)  DISTILL_ALPHA=$(DISTILL_ALPHA)"

# === Data Preparation ===

$(DATA_TSV): prepare_data.py
	@echo "📥 Preparing Newsroom dataset..."
	@mkdir -p $(DATA_DIR)
	$(PYTHON) prepare_data.py --out $(DATA_TSV)
	@echo "✅ Created $(DATA_TSV)"

# === Teacher Embeddings ===

$(TEACHER_DIR)/teacher_summaries.npy: $(DATA_TSV) precompute_teacher.py
	@mkdir -p $(TEACHER_DIR)
	@echo "🎓 Precomputing teacher embeddings..."
	$(PYTHON) precompute_teacher.py \
		--tsv $(DATA_TSV) \
		--out_dir $(TEACHER_DIR) \
		--teacher_model $(TEACHER_MODEL)
	@echo "✅ Teacher embeddings saved to $(TEACHER_DIR)/"

# === Training ===

$(MODEL_PT): $(DATA_TSV) $(TEACHER_DIR)/teacher_summaries.npy train.py model.py
	@mkdir -p $(BUILD_DIR)
	@echo "🏋️  Training Asterisk model with knowledge distillation..."
	$(PYTHON) train.py \
		--data $(DATA_TSV) \
		--teacher-dir $(TEACHER_DIR) \
		--output $(MODEL_PT) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--lr $(LR) \
		--patience $(PATIENCE) \
		--val-split $(VAL_SPLIT) \
		--distill-alpha $(DISTILL_ALPHA)
	@echo "✅ Model saved to $(MODEL_PT)"

# === Export & Quantize ===

$(MODEL_ONNX): $(MODEL_PT) quantize.py
	@mkdir -p $(BUILD_DIR)
	@echo "📦 Exporting to ONNX..."
	$(PYTHON) -c "from quantize import export_to_onnx; export_to_onnx('$(MODEL_PT)', '$(MODEL_ONNX)')"

$(MODEL_SIMPLIFIED): $(MODEL_ONNX)
	@echo "🔧 Simplifying ONNX model..."
	$(PYTHON) -c "from quantize import simplify_onnx; simplify_onnx('$(MODEL_ONNX)', '$(MODEL_SIMPLIFIED)')"

$(MODEL_INT8): $(MODEL_SIMPLIFIED)
	@echo "🗜️  Quantizing to INT8..."
	$(PYTHON) -c "from quantize import quantize_onnx; quantize_onnx('$(MODEL_SIMPLIFIED)', '$(MODEL_INT8)')"
	@echo "✅ Quantized model saved to $(MODEL_INT8)"

# === Vendor (bundle for distribution) ===
vendor: $(MODEL_INT8) ## Bundle artifacts into dist/ (INT8 model + tokenizer)
	@mkdir -p $(DIST_DIR)
	@if [ ! -f "$(DIST_MODEL_INT8)" ]; then \
		echo "📦 Copying INT8 model to $(DIST_MODEL_INT8)"; \
		cp $(MODEL_INT8) $(DIST_MODEL_INT8); \
	else \
		echo "ℹ️  INT8 model already present at $(DIST_MODEL_INT8), skipping copy"; \
	fi
	@if [ ! -f "$(TOKENIZER_DIR)/vocab.json" ]; then \
		echo "🔤 Vendoring tokenizer to $(TOKENIZER_DIR)"; \
		$(PYTHON) - <<-'PY'
	from transformers import GPT2Tokenizer
	import os
	os.makedirs("$(TOKENIZER_DIR)", exist_ok=True)
	tok = GPT2Tokenizer.from_pretrained("gpt2")
	tok.save_pretrained("$(TOKENIZER_DIR)")
	print("✅ Tokenizer saved to $(TOKENIZER_DIR)")
	PY
	else \
		echo "ℹ️  Tokenizer already present at $(TOKENIZER_DIR), skipping"; \
	fi