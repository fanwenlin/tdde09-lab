from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(init=True)
class Config:
    n_vocab: int = 50257
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        (batch_size, seq_len, n_embd) = x.shape
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        (batch_size, seq_len, n_embd) = x.shape
        head_embd = n_embd // self.n_head
        (q, k, v) = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_head, head_embd)
        k = k.view(batch_size, seq_len, self.n_head, head_embd)
        v = v.view(batch_size, seq_len, self.n_head, head_embd)
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        x = x.transpose(-2, -3).contiguous()
        x = x.view(batch_size, seq_len, n_embd)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def make_positions(n):
    return torch.arange(n, dtype=torch.long)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.n_vocab, config.n_embd)
        self.wpe = nn.Embedding(config.n_ctx, config.n_embd)
        self.h = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.n_vocab, bias=False)
        self.register_buffer("pos", make_positions(config.n_ctx), persistent=False)

    def forward(self, x):
        (batch_size, seq_len) = x.shape
        wte = self.wte(x)
        wpe = self.wpe(self.pos[:seq_len])
        x = wte + wpe
        x = self.h(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x
