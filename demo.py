"""
Rank sentences by semantic similarity using the Asterisk embedding model.

Loads sentences from a TSV pair file, embeds a random query, and returns
the top-k most similar sentences from the corpus. Includes latency benchmarking.
"""
import argparse
import random
import time
import numpy as np
import onnxruntime as ort
from transformers import GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import csv


def load_model(model_path: str):
    """Load tokenizer and ONNX model."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'mask_token': '[MASK]'})
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return tokenizer, session


def embed(text: str, tokenizer, session, max_len: int = 128):
    """Embed a single text string."""
    tokens = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=max_len)
    return session.run(None, {"input_ids": tokens["input_ids"]})[0].squeeze()


def load_sentences(path: str, sample_size: int = 1000):
    """Load unique sentences from a TSV pair file."""
    seen = set()
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) == 2:
                seen.add(row[0])
                seen.add(row[1])
    return random.sample(list(seen), min(sample_size, len(seen)))


def rank_similar(query: str, candidates: list, tokenizer, session, top_k: int = 10):
    """Rank candidates by cosine similarity to query."""
    query_vec = embed(query, tokenizer, session).reshape(1, -1)
    cand_vecs = np.stack([embed(c, tokenizer, session) for c in candidates])
    sims = cosine_similarity(query_vec, cand_vecs)[0]
    ranked = sorted(zip(candidates, sims), key=lambda x: -x[1])
    return ranked[:top_k]


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
    parser = argparse.ArgumentParser(description="Rank sentences by Asterisk embedding similarity")
    parser.add_argument("--data", default="data.tsv", help="Path to TSV pair file")
    parser.add_argument("--model", default="model_int8.onnx", help="Path to INT8 quantized ONNX model")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of sentences to sample")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top results to show")
    parser.add_argument("--query", type=str, default=None, help="Custom query (random if not provided)")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--benchmark-runs", type=int, default=100, help="Number of benchmark iterations")
    args = parser.parse_args()

    print(f"📦 Loading model: {args.model}")
    tokenizer, session = load_model(args.model)

    if args.benchmark:
        benchmark(tokenizer, session, runs=args.benchmark_runs)
        print()

    candidates = load_sentences(args.data, args.sample_size)
    query = args.query if args.query else random.choice(candidates)
    print(f"🔍 Query: {query}\n")

    others = [s for s in candidates if s != query]
    top = rank_similar(query, others, tokenizer, session, top_k=args.top_k)
    for i, (text, score) in enumerate(top, 1):
        print(f"{i:2d}. ({score:.4f}) {text}")


if __name__ == "__main__":
    main()

