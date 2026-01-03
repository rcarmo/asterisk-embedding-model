"""Asterisk Embedding Model.

Based on:
    Semenov, A. (2024). Asterisk*: Keep it Simple. arXiv:2411.05691.
    https://arxiv.org/abs/2411.05691
"""
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

class AsteriskEmbeddingModel(nn.Module):
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
        x = x.mean(dim=1)  # Mean pooling
        return self.proj(x)

# Example usage
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'mask_token': '[MASK]'})

model = AsteriskEmbeddingModel()
text = "This is a short summary."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
embedding = model(inputs["input_ids"])
print(embedding.shape)  # torch.Size([1, 256])

