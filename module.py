import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super(SingleHeadAttention, self).__init__()
        self.query = nn.Linear(config.n_embd, config.head_size)
        self.key = nn.Linear(config.n_embd, config.head_size)
        self.value = nn.Linear(config.n_embd, config.head_size)

        self.register_buffer(
            'attention_mask',
            torch.tril(torch.ones(config.max_seq, config.max_seq))
        )   # reduce memory usage by using a buffer

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q = self.query(x)   # (batch_size, seq_len, head_size)
        k = self.key(x) # (batch_size, seq_len, head_size)
        v = self.value(x)   # (batch_size, seq_len, head_size)

        weights = q @ k.transpose(-2, -1)    # (batch_size, seq_len, seq_len)
        weights = weights.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0,
            float('-inf')
        )
        weights = F.softmax(weights / math.sqrt(k.size(-1)), dim=-1)  # (batch_size, seq_len, seq_len)
        weights = self.dropout(weights)
        
        output = weights @ v  # (batch_size, seq_len, head_size)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [SingleHeadAttention(config) for _ in range(config.n_head)]
        )
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )
        x = self.proj(x)  # (batch_size, seq_len, hidden_dim)
        x = self.dropout(x)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.att = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x