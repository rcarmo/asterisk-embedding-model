"""Training script for Asterisk Embedding Model with Knowledge Distillation.

Based on:
    Semenov, A. (2024). Asterisk*: Keep it Simple. arXiv:2411.05691.
    https://arxiv.org/abs/2411.05691

The training combines:
1. Contrastive loss (InfoNCE) between sentence pairs
2. Knowledge distillation loss (MSE) aligning student embeddings with teacher embeddings
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from transformers import GPT2Tokenizer
from tqdm import tqdm

from model import AsteriskEmbeddingModel

# --- Config ---
SEED = 42

def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_lines(path):
    """Count lines in a file without loading it all into memory."""
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def get_line_offsets(path):
    """Build an index of byte offsets for each line (for random access)."""
    offsets = []
    with open(path, "rb") as f:
        offset = 0
        for line in f:
            offsets.append(offset)
            offset += len(line)
    return offsets


# --- Dataset ---
class SentencePairDataset(Dataset):
    """Memory-efficient dataset using line offsets for random access.
    
    Only stores byte offsets, not the actual text data.
    """
    def __init__(self, path, tokenizer, max_len=128):
        self.path = path
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Build line offset index (small memory footprint)
        print(f"Building line index for {path}...")
        self.offsets = get_line_offsets(path)
        self.length = len(self.offsets)
        print(f"  Indexed {self.length:,} rows")

    def __len__(self):
        return self.length

    def _read_line(self, idx):
        """Read a single line by seeking to its offset."""
        with open(self.path, "r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            line = f.readline().rstrip("\n")
        return line

    def __getitem__(self, idx):
        line = self._read_line(idx)
        parts = line.split("\t", 1)
        if len(parts) != 2:
            # Return empty pair for malformed lines
            s1, s2 = "", ""
        else:
            s1, s2 = parts
        
        t1 = self.tokenizer(s1, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_len)
        t2 = self.tokenizer(s2, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_len)
        return t1["input_ids"].squeeze(0), t2["input_ids"].squeeze(0)


class DistillationDataset(Dataset):
    """Memory-efficient dataset for knowledge distillation.
    
    Uses memory-mapped arrays for teacher embeddings and line offsets for text.
    """
    def __init__(self, tsv_path, teacher_dir, tokenizer, max_len=128):
        teacher_dir = Path(teacher_dir)
        self.tsv_path = tsv_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Load metadata
        meta_path = teacher_dir / "teacher_meta.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        self.num_rows = meta["num_rows"]
        self.teacher_dim = meta["dim"]
        
        # Load teacher embeddings as memory-mapped arrays (lazy loading)
        self.teacher_s = np.memmap(
            teacher_dir / "teacher_summaries.npy", 
            dtype=np.float16, mode="r", shape=(self.num_rows, self.teacher_dim)
        )
        self.teacher_a = np.memmap(
            teacher_dir / "teacher_articles.npy",
            dtype=np.float16, mode="r", shape=(self.num_rows, self.teacher_dim)
        )
        
        # Build line offset index instead of loading all text
        print(f"Building line index for {tsv_path}...")
        self.offsets = get_line_offsets(tsv_path)
        
        # Verify alignment
        assert len(self.offsets) == self.num_rows, \
            f"TSV rows ({len(self.offsets)}) != teacher embeddings ({self.num_rows})"
        print(f"  Indexed {self.num_rows:,} rows")

    def __len__(self):
        return self.num_rows

    def _read_line(self, idx):
        """Read a single line by seeking to its offset."""
        with open(self.tsv_path, "r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            line = f.readline().rstrip("\n")
        return line

    def __getitem__(self, idx):
        line = self._read_line(idx)
        parts = line.split("\t", 1)
        if len(parts) != 2:
            s1, s2 = "", ""
        else:
            s1, s2 = parts
        
        t1 = self.tokenizer(s1, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_len)
        t2 = self.tokenizer(s2, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_len)
        
        # Get teacher embeddings (convert to float32)
        teacher_emb1 = torch.from_numpy(self.teacher_s[idx].astype(np.float32))
        teacher_emb2 = torch.from_numpy(self.teacher_a[idx].astype(np.float32))
        
        return (
            t1["input_ids"].squeeze(0), 
            t2["input_ids"].squeeze(0),
            teacher_emb1,
            teacher_emb2
        )

# --- Contrastive Loss ---
def contrastive_loss(emb1, emb2, temperature=0.05):
    """Symmetric InfoNCE loss (bidirectional)."""
    emb1 = F.normalize(emb1, dim=1)
    emb2 = F.normalize(emb2, dim=1)
    sim = torch.matmul(emb1, emb2.T) / temperature
    labels = torch.arange(len(emb1)).to(emb1.device)
    # Bidirectional: both directions
    loss_1to2 = F.cross_entropy(sim, labels)
    loss_2to1 = F.cross_entropy(sim.T, labels)
    return (loss_1to2 + loss_2to1) / 2


# --- Distillation Loss ---
def distillation_loss(student_emb, teacher_emb):
    """MSE loss between normalized student and teacher embeddings.
    
    Note: Teacher embeddings are already normalized during precomputation.
    Student embeddings are normalized here to match.
    """
    student_emb = F.normalize(student_emb, dim=1)
    # Teacher may have different dim - project if needed via the model's projection layer
    return F.mse_loss(student_emb, teacher_emb)


# --- Combined Loss ---
def combined_loss(emb1, emb2, teacher_emb1, teacher_emb2, alpha=0.5, temperature=0.05):
    """Combined contrastive + distillation loss.
    
    Args:
        emb1, emb2: Student embeddings (256-dim)
        teacher_emb1, teacher_emb2: Teacher embeddings (384-dim or 768-dim)
        alpha: Weight for distillation loss (0 = pure contrastive, 1 = pure distillation)
        temperature: Temperature for contrastive loss
    
    Returns:
        Combined loss, contrastive component, distillation component
    """
    # Contrastive loss on student embeddings
    l_contrast = contrastive_loss(emb1, emb2, temperature)
    
    # Distillation loss - align with teacher
    # Note: dimensions may differ, so we compute cosine similarity loss instead of MSE
    emb1_norm = F.normalize(emb1, dim=1)
    emb2_norm = F.normalize(emb2, dim=1)
    teacher_emb1_norm = F.normalize(teacher_emb1, dim=1)
    teacher_emb2_norm = F.normalize(teacher_emb2, dim=1)
    
    # Cosine embedding loss: 1 - cos_sim (we want them aligned)
    l_distill1 = 1 - F.cosine_similarity(emb1_norm, teacher_emb1_norm).mean()
    l_distill2 = 1 - F.cosine_similarity(emb2_norm, teacher_emb2_norm).mean()
    l_distill = (l_distill1 + l_distill2) / 2
    
    loss = (1 - alpha) * l_contrast + alpha * l_distill
    return loss, l_contrast, l_distill


# --- Training Loop ---
def train(model, train_loader, val_loader, optimizer, scheduler, device, 
          epochs=5, patience=3, use_amp=True, distill_alpha=0.5, use_distillation=True):
    """Training loop with validation, early stopping, and mixed precision."""
    model.to(device)
    scaler = GradScaler(enabled=use_amp)
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        total_contrast = 0
        total_distill = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            if use_distillation:
                input1, input2, teacher1, teacher2 = batch
                input1, input2 = input1.to(device), input2.to(device)
                teacher1, teacher2 = teacher1.to(device), teacher2.to(device)
            else:
                input1, input2 = batch
                input1, input2 = input1.to(device), input2.to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=use_amp):
                emb1 = model(input1)
                emb2 = model(input2)
                
                if use_distillation:
                    loss, l_c, l_d = combined_loss(emb1, emb2, teacher1, teacher2, alpha=distill_alpha)
                    total_contrast += l_c.item()
                    total_distill += l_d.item()
                else:
                    loss = contrastive_loss(emb1, emb2)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        n_batches = len(train_loader)
        train_loss = total_loss / n_batches
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                if use_distillation:
                    input1, input2, teacher1, teacher2 = batch
                    input1, input2 = input1.to(device), input2.to(device)
                    teacher1, teacher2 = teacher1.to(device), teacher2.to(device)
                else:
                    input1, input2 = batch
                    input1, input2 = input1.to(device), input2.to(device)
                
                with autocast(enabled=use_amp):
                    emb1 = model(input1)
                    emb2 = model(input2)
                    if use_distillation:
                        loss, _, _ = combined_loss(emb1, emb2, teacher1, teacher2, alpha=distill_alpha)
                    else:
                        loss = contrastive_loss(emb1, emb2)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        # Logging
        if use_distillation:
            print(f"Epoch {epoch+1}: loss={train_loss:.4f} (contrast={total_contrast/n_batches:.4f}, "
                  f"distill={total_distill/n_batches:.4f}), val_loss={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
        else:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Train Asterisk Embedding Model")
    parser.add_argument("--data", type=str, default="data.tsv", help="Path to TSV dataset")
    parser.add_argument("--teacher-dir", type=str, default=None, help="Path to teacher embeddings directory")
    parser.add_argument("--output", type=str, default="model.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=5, help="Max training epochs (early stopping typically triggers in 2-4)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--distill-alpha", type=float, default=0.5, help="Distillation weight (0=contrastive only, 1=distill only)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available() and not args.no_amp
    
    # Check if we have teacher embeddings
    use_distillation = args.teacher_dir is not None and Path(args.teacher_dir).exists()
    if use_distillation:
        teacher_meta = Path(args.teacher_dir) / "teacher_meta.json"
        use_distillation = teacher_meta.exists()
    
    print(f"Device: {device}, AMP: {use_amp}, Distillation: {use_distillation}")
    if use_distillation:
        print(f"  Teacher dir: {args.teacher_dir}, alpha: {args.distill_alpha}")
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'mask_token': '[MASK]'})

    # Model
    model = AsteriskEmbeddingModel(vocab_size=len(tokenizer))

    # Dataset with train/val split
    if use_distillation:
        full_dataset = DistillationDataset(args.data, args.teacher_dir, tokenizer)
    else:
        full_dataset = SentencePairDataset(args.data, tokenizer)
    
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    num_workers = 4 if device == "cuda" else 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train
    model = train(model, train_loader, val_loader, optimizer, scheduler, device, 
                  epochs=args.epochs, patience=args.patience, use_amp=use_amp,
                  distill_alpha=args.distill_alpha, use_distillation=use_distillation)

    # Save model
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()