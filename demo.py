"""
Rank sentences by semantic similarity using the Asterisk embedding model.

Loads sentences from a TSV pair file, embeds a random query, and returns
the top-k most similar sentences from the corpus. Includes latency benchmarking.
"""
import argparse
import random
import time
import sys
import gc
import numpy as np
import onnxruntime as ort
from transformers import GPT2Tokenizer
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import csv

try:
    import psutil
except ImportError:  # fall back to stdlib
    psutil = None
import resource


def current_memory_mb() -> float:
    """Return current RSS memory usage in MB."""
    if psutil is not None:
        return psutil.Process().memory_info().rss / (1024 ** 2)
    # ru_maxrss is in KB on Linux, bytes on macOS; assume Linux here
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss_kb / 1024.0


def load_model(model_path: str):
    """Load tokenizer and ONNX model."""
    tokenizer = GPT2Tokenizer.from_pretrained("dist/tokenizer")
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'mask_token': '[MASK]'})
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return tokenizer, session


def embed(text: str, tokenizer, session, max_len: int = 128):
    """Embed a single text string."""
    tokens = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=max_len)
    return session.run(None, {"input_ids": tokens["input_ids"]})[0].squeeze()


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
            for s in row:  # two sentences per row
                if needle and needle not in s.lower():
                    continue
                n += 1
                if len(reservoir) < sample_size:
                    reservoir.append(s)
                else:
                    j = random.randrange(n)
                    if j < sample_size:
                        reservoir[j] = s
    # Explicitly free any lingering buffers
    del reader
    gc.collect()
    return reservoir


def rank_similar(query: str, candidates: list, tokenizer, session, top_k: int = 10):
    """Rank candidates by cosine similarity to query with streaming top-k to reduce memory."""
    query_vec = embed(query, tokenizer, session).reshape(1, -1)
    top_heap = []  # min-heap of (sim, text)
    for text in candidates:
        cand_vec = embed(text, tokenizer, session).reshape(1, -1)
        sim = cosine_similarity(query_vec, cand_vec)[0][0]
        if len(top_heap) < top_k:
            heapq.heappush(top_heap, (sim, text))
        else:
            if sim > top_heap[0][0]:
                heapq.heapreplace(top_heap, (sim, text))
    ranked = sorted(top_heap, key=lambda x: -x[0])
    return [(text, sim) for sim, text in ranked]


def benchmark(tokenizer, session, runs: int = 100):
    """Benchmark embedding latency."""
    text = "A short summary about climate change and its global impact."
    # Warmup
    for _ in range(10):
        embed(text, tokenizer, session)
    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        embed(text, tokenizer, session)
        times.append(time.perf_counter() - start)
    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    p95_ms = np.percentile(times, 95) * 1000
    print(f"⏱️  Latency ({runs} runs): {avg_ms:.2f} ± {std_ms:.2f} ms (p95: {p95_ms:.2f} ms)")


def main():
    # Capture baseline memory before loading anything
    baseline_mem_mb = current_memory_mb()

    parser = argparse.ArgumentParser(description="Rank sentences by Asterisk embedding similarity")
    parser.add_argument("--data", default="data/data.tsv", help="Path to TSV pair file (vendored)")
    parser.add_argument("--model", default="dist/model_int8.onnx", help="Path to INT8 quantized ONNX model (vendored)")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of sentences to sample")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results to show")
    parser.add_argument("--query", type=str, default=None, help="Custom query (random if not provided)")
    parser.add_argument("--contains", type=str, default=None, help="Filter sampled sentences to those containing this substring")
    parser.add_argument("--benchmark", type=int, default=0, help="Run latency benchmark for N iterations (0 to skip)")
    args = parser.parse_args()

    print(f"📦 Loading model: {args.model}")
    t0 = time.perf_counter()
    tokenizer, session = load_model(args.model)
    load_ms = (time.perf_counter() - t0) * 1000

    # If benchmarking only, skip ranking output
    if args.benchmark > 0:
        benchmark(tokenizer, session, runs=args.benchmark)
        mem_mb = current_memory_mb() - baseline_mem_mb
        print(f"⏱️  Timings: load={load_ms:.2f} ms (benchmark only), memory≈{mem_mb:.2f} MB")
        return

    candidates = load_sentences(args.data, args.sample_size, contains=args.contains)
    print(f"📊 Sample size: {len(candidates)} (requested {args.sample_size}), filter={args.contains}")
    # Choose query index to avoid duplicating list
    if args.query:
        query = args.query
        others = candidates
    else:
        qi = random.randrange(len(candidates)) if candidates else 0
        query = candidates.pop(qi)
        others = candidates
    print(f"🔍 Query: {query}\n")
    t1 = time.perf_counter()
    top = rank_similar(query, others, tokenizer, session, top_k=args.top_k)
    rank_ms = (time.perf_counter() - t1) * 1000
    total_embeds = len(others) + 1  # query + candidates
    per_embed_ms = rank_ms / total_embeds if total_embeds else 0
    del others  # free candidate list
    gc.collect()
    for i, (text, score) in enumerate(top, 1):
        print(f"{i:2d}. ({score:.4f}) {text}")

    mem_mb = current_memory_mb() - baseline_mem_mb
    print(f"\n⏱️  Timings: load={load_ms:.2f} ms, embed+rank={rank_ms:.2f} ms (per-embed≈{per_embed_ms:.4f} ms), memory≈{mem_mb:.2f} MB")


if __name__ == "__main__":
    main()

