# Asterisk Embedding Model

A compact, efficient text embedding model based on the Transformer encoder architecture.

**Paper:** Semenov, A. (2024). *Asterisk\*: Keep it Simple*. [arXiv:2411.05691](https://arxiv.org/abs/2411.05691)

## Overview

Asterisk is a lightweight sentence embedding model designed for:
- **Small footprint**: 2 layers, 2 attention heads, ~2.6M parameters
- **Fast inference**: INT8 quantized ONNX for CPU deployment
- **Knowledge distillation**: Learn from larger teacher models

| Component | Value |
|-----------|-------|
| Hidden size | 512 |
| Layers | 2 |
| Attention heads | 2 |
| Output dimension | 256 |
| Tokenizer | GPT-2 BPE |

## Quick Start

```bash
# Install dependencies
make install

# Run full pipeline (data → teacher → train → export)
make all

# Or step by step:
make data      # Prepare Newsroom dataset
make teacher   # Precompute teacher embeddings
make train     # Train with knowledge distillation
make export    # Export to ONNX + quantize to INT8

# Test the model
make benchmark # Run latency benchmark
make demo      # Similarity ranking demo
```

## Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. DATA PREPARATION                                                │
│     prepare_data.py                                                 │
│     └── data.tsv (summary, article pairs)                           │
├─────────────────────────────────────────────────────────────────────┤
│  2. TEACHER EMBEDDINGS (optional but recommended)                   │
│     precompute_teacher.py                                           │
│     └── teacher/teacher_summaries.npy, teacher_articles.npy         │
├─────────────────────────────────────────────────────────────────────┤
│  3. TRAINING                                                        │
│     train.py                                                        │
│     └── model.pt                                                    │
│     Loss = (1-α)·InfoNCE + α·Distillation                           │
├─────────────────────────────────────────────────────────────────────┤
│  4. EXPORT & QUANTIZE                                               │
│     quantize.py                                                     │
│     └── model.onnx → model_simplified.onnx → model_int8.onnx        │
└─────────────────────────────────────────────────────────────────────┘
```

## Configuration

Override defaults via environment or make arguments:

```bash
make train EPOCHS=20 BATCH_SIZE=64 LR=1e-4 DISTILL_ALPHA=0.7
```

| Variable | Default | Description |
|----------|---------|-------------|
| `EPOCHS` | 5 | Maximum training epochs |
| `BATCH_SIZE` | 32 | Training batch size |
| `LR` | 2e-4 | Learning rate |
| `PATIENCE` | 3 | Early stopping patience |
| `VAL_SPLIT` | 0.1 | Validation set fraction |
| `DISTILL_ALPHA` | 0.5 | Distillation weight (0=contrastive only, 1=distill only) |
| `TEACHER_MODEL` | sentence-transformers/all-MiniLM-L6-v2 | Teacher model for distillation |

## Usage

### Python (PyTorch)

```python
import torch
from model import AsteriskEmbeddingModel
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AsteriskEmbeddingModel(vocab_size=len(tokenizer))
model.load_state_dict(torch.load("model.pt"))
model.eval()

text = "This is a sample sentence."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    embedding = model(inputs["input_ids"])  # [1, 256]
```

### ONNX Runtime (Production)

```python
import onnxruntime as ort
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

session = ort.InferenceSession("model_int8.onnx", providers=["CPUExecutionProvider"])

text = "This is a sample sentence."
tokens = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
embedding = session.run(None, {"input_ids": tokens["input_ids"]})[0]  # [1, 256]
```

## Files

| File | Description |
|------|-------------|
| `model.py` | Model architecture definition |
| `train.py` | Training script with distillation |
| `precompute_teacher.py` | Generate teacher embeddings |
| `prepare_data.py` | Prepare Newsroom training data |
| `quantize.py` | Export to ONNX and quantize |
| `inference.py` | PyTorch inference example |
| `demo.py` | Similarity search demo + benchmark |

## Memory Efficiency

The training pipeline is designed for large datasets:

- **Line-offset indexing**: TSV files are not loaded into memory; only byte offsets are stored
- **Memory-mapped teacher embeddings**: `.npy` files are accessed via `np.memmap`
- **Streaming data preparation**: Newsroom dataset is processed in streaming mode

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{semenov2024asterisk,
  title={Asterisk*: Keep it Simple},
  author={Semenov, Andrew},
  journal={arXiv preprint arXiv:2411.05691},
  year={2024}
}
```
