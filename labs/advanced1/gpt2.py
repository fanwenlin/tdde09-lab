from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    n_vocab = 30522
    n_ctx = 512
    n_embd = 768
    n_head = 12
    n_layer = 12
    n_type_vocab_size = 2


# def gelu(x):
#     return 0.5 * x * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x**3)))
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / (2.0 ** 0.5)))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape
        x = self.c_fc(x)
        x = gelu(x)
        x = self.c_proj(x)
        return x


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # self.register_buffer("mask", make_causal_mask(config.n_ctx), persistent=False)

    def forward(self, x, attention_mask):
        batch_size, seq_len, n_embd = x.shape
        head_embd = n_embd // self.n_head
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_head, head_embd)
        k = k.view(batch_size, seq_len, self.n_head, head_embd)
        v = v.view(batch_size, seq_len, self.n_head, head_embd)
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        x = q @ k.transpose(-1, -2)
        x = x / head_embd**0.5
        if attention_mask is not None:
            # [B, S] -> [B, 1, 1, S], 1 keeps token, 0 masks token
            key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            x = x.masked_fill(~key_mask, torch.finfo(x.dtype).min)
        x = torch.softmax(x, dim=-1)
        x = x @ v
        x = x.transpose(-2, -3).contiguous()
        x = x.view(batch_size, seq_len, n_embd)
        x = self.c_proj(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.g = nn.Parameter(torch.ones(config.n_embd))
        self.b = nn.Parameter(torch.zeros(config.n_embd))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(unbiased=False, dim=-1, keepdim=True)
        return self.g * (x - mean) / torch.sqrt(variance + 1e-12) + self.b


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config)
        self.attn = Attention(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask):
        x = self.ln_1(x+self.attn(x, attention_mask))
        x = self.ln_2(x+self.mlp(x))
        return x


def make_positions(n):
    return torch.arange(n, dtype=torch.long)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.n_vocab, config.n_embd)
        self.wpe = nn.Embedding(config.n_ctx, config.n_embd)
        self.type_token_wpe = nn.Embedding(config.n_type_vocab_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config)
        # self.lm_head = nn.Linear(config.n_embd, config.n_vocab, bias=False)
        self.cls_pooler = nn.Linear(config.n_embd, config.n_embd)
        self.cls_pooler_activation = nn.Tanh()
        self.register_buffer("pos", make_positions(config.n_ctx), persistent=False)

    def forward(self, x=None, token_type_ids=None, attention_mask=None, input_ids=None):
        if x is None:
            x = input_ids
        batch_size, seq_len = x.shape
        wte = self.wte(x)
        wpe = self.wpe(self.pos[:seq_len])

        if attention_mask is None:
            attention_mask = torch.ones_like(x)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(x)

        type_token_wte = self.type_token_wpe(token_type_ids)
        x = wte + wpe + type_token_wte
        x = self.ln_f(x)
        for h in self.h:
            x = h(x, attention_mask=attention_mask)
        pooled = self.cls_pooler_activation(self.cls_pooler(x[:, 0]))
        return (x, pooled)
