"""Inference script for Asterisk Embedding Model.

Based on:
    Semenov, A. (2024). Asterisk*: Keep it Simple. arXiv:2411.05691.
    https://arxiv.org/abs/2411.05691
"""
import torch
from time import time
from transformers import GPT2Tokenizer
from model import AsteriskEmbeddingModel

def load_model(path="model.pt"):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'mask_token': '[MASK]'})
    model = AsteriskEmbeddingModel(vocab_size=len(tokenizer))
    model.token_emb = torch.nn.Embedding(len(tokenizer), 512)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model, tokenizer

def embed(text, model, tokenizer, max_len=128):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
    with torch.no_grad():
        return model(inputs["input_ids"]).squeeze().numpy()

if __name__ == "__main__":
    model, tokenizer = load_model()
    x = time()
    text = "A short summary about climate change."

    vec = embed(text, model, tokenizer)
    print(time()-x)
    #print(f"Embedding shape: {vec.shape}\n{vec}")

