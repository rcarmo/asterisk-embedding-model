# precompute_teacher_embeddings.py
# pip install torch transformers tqdm numpy

import json
import math
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def count_lines(path):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n

def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1e-6)
    return summed / lengths

def main(tsv_path, out_dir, teacher_model, batch_size=128, fp16=True, device="cuda"):
    tsv_path = Path(tsv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Counting lines...")
    n = count_lines(tsv_path)
    print(f"Found {n} rows")

    print("Loading teacher tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model)
    model = AutoModel.from_pretrained(teacher_model).to(device)
    model.eval()

    # warmup to get dim
    with torch.no_grad():
        sample = tokenizer("hello world", return_tensors="pt", truncation=True, padding=True).to(device)
        out = model(**sample, return_dict=True)
        dim = out.last_hidden_state.size(-1)

    dtype = np.float16
    s_path = out_dir / "teacher_summaries.npy"
    a_path = out_dir / "teacher_articles.npy"

    # create memmaps
    s_mem = np.memmap(s_path, dtype=dtype, mode="w+", shape=(n, dim))
    a_mem = np.memmap(a_path, dtype=dtype, mode="w+", shape=(n, dim))

    # stream and encode
    idx = 0
    with open(tsv_path, "r", encoding="utf-8") as f:
        batch_s = []
        batch_a = []
        for line in tqdm(f, total=n, desc="reading TSV"):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                s, a = line.split("\t", 1)
            except ValueError:
                s, a = "", ""
            batch_s.append(s)
            batch_a.append(a)
            if len(batch_s) >= batch_size:
                # encode summaries
                with torch.no_grad():
                    tok_s = tokenizer(batch_s, truncation=True, padding=True, return_tensors="pt").to(device)
                    out_s = model(**tok_s, return_dict=True)
                    pooled_s = mean_pool(out_s.last_hidden_state, tok_s["attention_mask"])
                    pooled_s = F.normalize(pooled_s, dim=-1).cpu().numpy().astype(dtype)

                    tok_a = tokenizer(batch_a, truncation=True, padding=True, return_tensors="pt").to(device)
                    out_a = model(**tok_a, return_dict=True)
                    pooled_a = mean_pool(out_a.last_hidden_state, tok_a["attention_mask"])
                    pooled_a = F.normalize(pooled_a, dim=-1).cpu().numpy().astype(dtype)

                b = pooled_s.shape[0]
                s_mem[idx:idx+b, :] = pooled_s
                a_mem[idx:idx+b, :] = pooled_a
                idx += b
                batch_s = []
                batch_a = []

        # final partial batch
        if batch_s:
            with torch.no_grad():
                tok_s = tokenizer(batch_s, truncation=True, padding=True, return_tensors="pt").to(device)
                out_s = model(**tok_s, return_dict=True)
                pooled_s = mean_pool(out_s.last_hidden_state, tok_s["attention_mask"])
                pooled_s = F.normalize(pooled_s, dim=-1).cpu().numpy().astype(dtype)

                tok_a = tokenizer(batch_a, truncation=True, padding=True, return_tensors="pt").to(device)
                out_a = model(**tok_a, return_dict=True)
                pooled_a = mean_pool(out_a.last_hidden_state, tok_a["attention_mask"])
                pooled_a = F.normalize(pooled_a, dim=-1).cpu().numpy().astype(dtype)

            b = pooled_s.shape[0]
            s_mem[idx:idx+b, :] = pooled_s
            a_mem[idx:idx+b, :] = pooled_a
            idx += b

    # flush memmaps
    s_mem.flush()
    a_mem.flush()

    meta = {"num_rows": n, "dim": dim, "dtype": str(dtype), "teacher_model": teacher_model}
    with open(out_dir / "teacher_meta.json", "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

    print("Done. Saved teacher embeddings to", out_dir)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--teacher_model", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    main(args.tsv, args.out_dir, args.teacher_model, batch_size=args.batch_size, fp16=args.fp16, device=args.device)

