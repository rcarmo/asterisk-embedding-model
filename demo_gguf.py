#!/usr/bin/env python
"""
Demo for GGUF model inference.

Loads the Asterisk embedding model from GGUF format and runs similarity ranking.
Note: GGUF is a storage format; we reconstruct the PyTorch model from GGUF weights.
"""
import argparse
import random
import time
import sys
import gc
import csv
import heapq
import numpy as np
from pathlib import Path

try:
    import gguf
except ImportError:
    sys.stderr.write("gguf not installed. Install with: pip install git+https://github.com/ggerganov/llama.cpp#subdirectory=gguf-py\n")
    sys.exit(1)

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import psutil
except ImportError:
    psutil = None
import resource


def current_memory_mb() -> float:
    """Return current RSS memory usage in MB."""
    if psutil is not None:
        return psutil.Process().memory_info().rss / (1024 ** 2)
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss_kb / 1024.0


class AsteriskEmbeddingModel(nn.Module):
    """Asterisk embedding model (reconstructed for GGUF inference)."""
    def __init__(self, vocab_size=50259, hidden_size=512, num_layers=2, num_heads=2, output_dim=256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(512, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(hidden_size, output_dim)

    def forward(self, input_ids):
        B, T = input_ids.size()
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.proj(x)


# GGUF block sizes for quantized formats
GGUF_BLOCK_SIZE = 32  # Q8_0, Q4_0, Q4_1, Q5_0, Q5_1 all use 32-element blocks


def dequantize_q8_0(data: np.ndarray, shape: tuple) -> np.ndarray:
    """Dequantize Q8_0 format: each block has 1 fp16 scale + 32 int8 values."""
    n_elements = int(np.prod(shape))
    n_blocks = (n_elements + GGUF_BLOCK_SIZE - 1) // GGUF_BLOCK_SIZE
    
    # Q8_0 block: 2 bytes (fp16 scale) + 32 bytes (int8 values) = 34 bytes
    block_size_bytes = 2 + GGUF_BLOCK_SIZE
    raw = data.view(np.uint8).reshape(n_blocks, block_size_bytes)
    
    # Extract scales (first 2 bytes as fp16)
    scales = raw[:, :2].view(np.float16).astype(np.float32)
    # Extract quantized values (remaining 32 bytes as int8)
    quants = raw[:, 2:].view(np.int8).astype(np.float32)
    
    # Dequantize: value = scale * quant
    dequantized = (scales * quants).flatten()[:n_elements]
    return dequantized.astype(np.float32)


def dequantize_q4_0(data: np.ndarray, shape: tuple) -> np.ndarray:
    """Dequantize Q4_0 format: each block has 1 fp16 scale + 16 bytes (32 4-bit values)."""
    n_elements = int(np.prod(shape))
    n_blocks = (n_elements + GGUF_BLOCK_SIZE - 1) // GGUF_BLOCK_SIZE
    
    # Q4_0 block: 2 bytes (fp16 scale) + 16 bytes (32 4-bit values packed) = 18 bytes
    block_size_bytes = 2 + GGUF_BLOCK_SIZE // 2
    raw = data.view(np.uint8).reshape(n_blocks, block_size_bytes)
    
    # Extract scales
    scales = raw[:, :2].view(np.float16).astype(np.float32)
    # Extract packed 4-bit values
    packed = raw[:, 2:]
    
    # Unpack: low 4 bits and high 4 bits
    low = (packed & 0x0F).astype(np.int8) - 8  # Q4_0 uses unsigned with -8 offset
    high = ((packed >> 4) & 0x0F).astype(np.int8) - 8
    quants = np.stack([low, high], axis=-1).reshape(n_blocks, GGUF_BLOCK_SIZE).astype(np.float32)
    
    dequantized = (scales * quants).flatten()[:n_elements]
    return dequantized.astype(np.float32)


def dequantize_tensor(tensor, shape: tuple) -> np.ndarray:
    """Dequantize a GGUF tensor based on its type."""
    qtype = tensor.tensor_type
    data = np.array(tensor.data, copy=True)
    
    # GGMLQuantizationType values
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    
    if qtype == F32:
        return data.reshape(shape).astype(np.float32)
    elif qtype == F16:
        return data.view(np.float16).reshape(shape).astype(np.float32)
    elif qtype == Q8_0:
        return dequantize_q8_0(data, shape).reshape(shape)
    elif qtype == Q4_0:
        return dequantize_q4_0(data, shape).reshape(shape)
    else:
        # Fallback: try to interpret as float32
        print(f"⚠️  Unknown quantization type {qtype} for tensor, attempting float32")
        return data.astype(np.float32).reshape(shape)


def load_gguf_model(gguf_path: str):
    """Load model weights from GGUF file and reconstruct PyTorch model."""
    reader = gguf.GGUFReader(gguf_path)
    
    # Extract metadata - values are in parts[-1] for uint32 fields
    metadata = {}
    shape_metadata = {}  # For quantized tensor shapes
    for field in reader.fields.values():
        if field.name.startswith("asterisk.shape."):
            # Shape metadata stored as comma-separated string
            tensor_name = field.name.replace("asterisk.shape.", "")
            # String values are in parts[4] for string fields
            if len(field.parts) >= 5:
                shape_str = bytes(field.parts[-1]).decode('utf-8')
                shape_metadata[tensor_name] = tuple(int(d) for d in shape_str.split(","))
        elif field.name.startswith("asterisk."):
            key = field.name.replace("asterisk.", "")
            # The actual value is in parts[-1] for typed fields
            if len(field.parts) >= 4:
                metadata[key] = int(field.parts[-1][0])
    
    hidden_size = metadata.get("hidden_size", 512)
    num_layers = metadata.get("layers", 2)
    num_heads = metadata.get("heads", 2)
    output_dim = metadata.get("output_dim", 256)
    
    # Create model
    model = AsteriskEmbeddingModel(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        output_dim=output_dim
    )
    
    # Load tensors with dequantization support
    state_dict = {}
    for tensor in reader.tensors:
        name = tensor.name
        # Use shape from metadata for quantized tensors, otherwise from GGUF
        if name in shape_metadata:
            # Quantized tensor: metadata has correct PyTorch shape
            shape = shape_metadata[name]
            data = dequantize_tensor(tensor, shape)
            # No transpose needed - metadata has PyTorch shape
        else:
            # Non-quantized tensor: GGUF stores [cols, rows]
            shape = tuple(tensor.shape)
            data = dequantize_tensor(tensor, shape)
            # GGUF stores 2D as [cols, rows], transpose for PyTorch
            if len(shape) == 2:
                data = data.T
        
        state_dict[name] = torch.from_numpy(data.copy())
    
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_model(gguf_path: str, tokenizer_path: str):
    """Load tokenizer and GGUF model."""
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'mask_token': '[MASK]'})
    model = load_gguf_model(gguf_path)
    return tokenizer, model


@torch.no_grad()
def embed(text: str, tokenizer, model, max_len: int = 128):
    """Embed a single text string."""
    tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
    return model(tokens["input_ids"]).squeeze().numpy()


def load_sentences(path: str, sample_size: int = 1000, contains: str | None = None):
    """Reservoir-sample sentences from TSV; optional substring filter."""
    reservoir = []
    n = 0
    needle = contains.lower() if contains else None
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) != 2:
                continue
            for s in row:
                if needle and needle not in s.lower():
                    continue
                n += 1
                if len(reservoir) < sample_size:
                    reservoir.append(s)
                else:
                    j = random.randrange(n)
                    if j < sample_size:
                        reservoir[j] = s
    del reader
    gc.collect()
    return reservoir


def rank_similar(query: str, candidates: list, tokenizer, model, top_k: int = 10):
    """Rank candidates by cosine similarity to query."""
    query_vec = embed(query, tokenizer, model).reshape(1, -1)
    top_heap = []
    for text in candidates:
        cand_vec = embed(text, tokenizer, model).reshape(1, -1)
        sim = cosine_similarity(query_vec, cand_vec)[0][0]
        if len(top_heap) < top_k:
            heapq.heappush(top_heap, (sim, text))
        elif sim > top_heap[0][0]:
            heapq.heapreplace(top_heap, (sim, text))
    ranked = sorted(top_heap, key=lambda x: -x[0])
    return [(text, sim) for sim, text in ranked]


def benchmark(tokenizer, model, runs: int = 100):
    """Benchmark embedding latency."""
    text = "A short summary about climate change and its global impact."
    for _ in range(10):
        embed(text, tokenizer, model)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        embed(text, tokenizer, model)
        times.append(time.perf_counter() - start)
    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    p95_ms = np.percentile(times, 95) * 1000
    print(f"⏱️  Latency ({runs} runs): {avg_ms:.2f} ± {std_ms:.2f} ms (p95: {p95_ms:.2f} ms)")


def main():
    # Capture baseline memory before loading anything
    baseline_mem_mb = current_memory_mb()

    parser = argparse.ArgumentParser(description="GGUF model similarity demo")
    parser.add_argument("--model", default="dist/model_q8_0.gguf", help="Path to GGUF model")
    parser.add_argument("--tokenizer", default="dist/tokenizer", help="Path to tokenizer directory")
    parser.add_argument("--data", default="data/data.tsv", help="Path to TSV pair file")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of sentences to sample")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results to show")
    parser.add_argument("--query", type=str, default=None, help="Custom query (random if not provided)")
    parser.add_argument("--contains", type=str, default=None, help="Filter sentences containing substring")
    parser.add_argument("--benchmark", type=int, default=0, help="Run latency benchmark for N iterations")
    args = parser.parse_args()

    print(f"📦 Loading GGUF model: {args.model}")
    t0 = time.perf_counter()
    tokenizer, model = load_model(args.model, args.tokenizer)
    load_ms = (time.perf_counter() - t0) * 1000

    if args.benchmark > 0:
        benchmark(tokenizer, model, runs=args.benchmark)
        mem_mb = current_memory_mb() - baseline_mem_mb
        print(f"⏱️  Timings: load={load_ms:.2f} ms (benchmark only), memory≈{mem_mb:.2f} MB")
        return

    candidates = load_sentences(args.data, args.sample_size, contains=args.contains)
    print(f"📊 Sample size: {len(candidates)} (requested {args.sample_size}), filter={args.contains}")
    
    if args.query:
        query = args.query
        others = candidates
    else:
        qi = random.randrange(len(candidates)) if candidates else 0
        query = candidates.pop(qi)
        others = candidates
    
    print(f"🔍 Query: {query}\n")
    t1 = time.perf_counter()
    top = rank_similar(query, others, tokenizer, model, top_k=args.top_k)
    rank_ms = (time.perf_counter() - t1) * 1000
    total_embeds = len(others) + 1
    per_embed_ms = rank_ms / total_embeds if total_embeds else 0
    
    del others
    gc.collect()
    
    for i, (text, score) in enumerate(top, 1):
        print(f"{i:2d}. ({score:.4f}) {text}")

    mem_mb = current_memory_mb() - baseline_mem_mb
    print(f"\n⏱️  Timings: load={load_ms:.2f} ms, embed+rank={rank_ms:.2f} ms (per-embed≈{per_embed_ms:.4f} ms), memory≈{mem_mb:.2f} MB")


if __name__ == "__main__":
    main()
