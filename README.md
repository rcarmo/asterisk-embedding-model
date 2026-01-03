# Asterisk Embedding Model

![Asterisk Logo](docs/asterisk-256.png)

A compact, efficient 256-dimensional text embedding model based on a Transformer encoder.

**Paper:** Semenov, A. (2024). *Asterisk\*: Keep it Simple*. [arXiv:2411.05691](https://arxiv.org/abs/2411.05691)

## Why

I wanted a sentence embedding model with:
- A smaller-dimensional output that still preserves semantic similarity across tens of thousands of segments
- Low resource requirements for edge devices (~30MB on disk; CPU-friendly)
- Fast enough for real-time applications
- A low memory footprint during inference (*still a work in progress*; ONNX Runtime + GPT-2 tokenizer currently uses >700MB RAM, so I'm exploring GGUF and TFlite as alternatives, although they have their own challenges)
- Relatively fast training even on modest consumer-grade hardware (around 2-3h on a 12GB RTX3060 for the NEWSROOM dataset)

The key application I had in mind was semantic search over low thousands of news summaries on a CPU, with latency under 50ms per query, as [my RSS feed summarizer](https://github.com/rcarmo/feed-summarizer) was hitting a brick wall Simhash and FTS5-based approaches and I didn't want to rely on an external embedding service.

I also need it for clustering or deduplication of personal notes and blog posts, again on low-resource hardware or at high speed on a laptop CPU.

Finally, out of curiosity: most literature focuses on larger, higher-accuracy models; I wanted to explore the other end of the spectrum—how simple can we go while staying useful?

Semenov's [*Asterisk\** paper](https://arxiv.org/abs/2411.05691) provided a solid starting point. This is by no means optimized or state-of-the-art, but I think it strikes a decent balance between size, speed, and performance for a few practical use cases.

## Demo

```bash
❯ make demo
ℹ️  INT8 model already present at dist/model_int8.onnx, skipping copy
🔍 Running similarity demo...
📦 Loading model: dist/model_int8.onnx
📊 Sample size: 100 (requested 100), filter=None
🔍 Query: Drivers beware: flurries beginning Sunday afternoon will blanket Ottawa with 10 to 15 cm of fresh snow by morning, according to Environment Canada.

 1. (0.7105) Looking for the ultimate flag for Canada Day? The Peace Tower flag could be all yours. Any Canadian living in the country can order one. But you won't receive it in time for this year's July 1st celebrations-or, frankly, anytime soon.
 2. (0.6511) Liberal Leader Justin Trudeau says today's surprise announcement by the Bank of Canada to cut its benchmark interest rate is further proof the Conservatives are bad managers of the economy.
 3. (0.5962) NEW ORLEANS — The huge storm headed for the Gulf Coast was upgraded to a Category 1 hurricane on Tuesday as it continued to gain strength, though when Hurricane Isaac makes landfall it will almost certainly be a far smaller storm than initially feared, forecasters said.
 4. (0.5679) A fire at a suburban Chicago air traffic control facility Friday morning halted all flights in and out of the city's two airports, threatening to send delays and cancelations rippling around the nation's air travel network.
 5. (0.4949) By LAWRENCE K. ALTMAN, MDAPRIL 28, 2014

⏱️  Timings: load=125.40 ms, embed+rank=370.29 ms (per-embed≈3.7029 ms), memory≈86.39 MB
```

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
| Training data | NEWSROOM dataset (1.3M article-summary pairs) |

### About the NEWSROOM dataset

The model in `dist` was trained on the NEWSROOM summarization dataset (Grusky et al., 2018): ~1.3M article–summary pairs from 38 major news publishers.

In this project I decided to use a Hugging Face mirror (`LogeshChandran/newsroom`), cleaning text and pairing each summary with the first paragraph of its source article. The prepared 350MB TSV lives in `data/data.tsv` after `make data`.

## Quick Start

First of all, ensure you're using `uv` and Python 3.12 (I have been trying to get TFlite support working, but it's pretty fragile across Python versions, so all we have now is ONNX + GGUF):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
# currently not recommended
# uv pip install -r requirements-tflite.txt
```

Then run the full pipeline:

```bash
# Install dependencies
make install

# Run full pipeline (data → teacher → train → export → vendor)
make all

# Or step by step:
make data      # Prepare Newsroom dataset → data/data.tsv
make teacher   # Precompute teacher embeddings → build/teacher/
make train     # Train with knowledge distillation → build/model.pt
make export    # Export to ONNX + quantize to INT8 → build/model_int8.onnx
make vendor    # Bundle dist/: model_int8.onnx + tokenizer/
make tflite    # Export TFLite model → dist/model.tflite

# Test the model (uses dist/model_int8.onnx)
make benchmark # Run latency benchmark
make demo      # Similarity ranking demo
```

## Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│ 1) DATA PREPARATION                                          │
│    prepare_data.py → data/data.tsv (summary, article pairs)  │
├──────────────────────────────────────────────────────────────┤
│ 2) TEACHER EMBEDDINGS (optional)                             │
│    precompute_teacher.py → build/teacher/*.npy, *.json       │
├──────────────────────────────────────────────────────────────┤
│ 3) TRAINING                                                  │
│    train.py → build/model.pt                                 │
│    Loss = (1-α)·InfoNCE + α·Distillation                     │
├──────────────────────────────────────────────────────────────┤
│ 4) EXPORT & QUANTIZE                                         │
│    quantize.py → build/model.onnx → model_simplified.onnx →  │
│                 build/model_int8.onnx                        │
├──────────────────────────────────────────────────────────────┤
│ 5) VENDOR                                                    │
│    make vendor → dist/model_int8.onnx, dist/tokenizer/       │
├──────────────────────────────────────────────────────────────┤
│ 6) TFLITE (optional)                                         │
│    make tflite → dist/model.tflite                           │
└──────────────────────────────────────────────────────────────┘
```

## Configuration

Override defaults via environment or make arguments:

```bash
make train EPOCHS=5 BATCH_SIZE=32 LR=1e-4 DISTILL_ALPHA=0.7
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

tokenizer = GPT2Tokenizer.from_pretrained("dist/tokenizer")  # after make vendor
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AsteriskEmbeddingModel(vocab_size=len(tokenizer))
model.load_state_dict(torch.load("build/model.pt", map_location="cpu"))
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

tokenizer = GPT2Tokenizer.from_pretrained("dist/tokenizer")  # after make vendor
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

session = ort.InferenceSession("dist/model_int8.onnx", providers=["CPUExecutionProvider"])

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
| `data/` | Working data directory (data.tsv) |
| `build/` | Working models + teacher embeddings |
| `dist/` | Distributed artifacts (model_int8.onnx, tokenizer/) |

## Memory Efficiency

The training pipeline is designed for large datasets:

- **Line-offset indexing**: TSV files are not loaded into memory; only byte offsets are stored
- **Memory-mapped teacher embeddings**: `.npy` files are accessed via `np.memmap`
- **Streaming data preparation**: Newsroom dataset is processed in streaming mode

The actual memory efficiency in production will depend on the peculiarities of your ONNX runtime and tokenizer implementation.

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
