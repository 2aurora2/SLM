import torch
import torch.nn as nn
import torch.nn.functional as F

from module import *

class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.max_seq, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_final = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, ids, targets=None):
        batch_size, seq_len = ids.size()
        token_emb = self.token_embedding(ids)
        pos_emb = self.position_embedding(
            torch.arange(seq_len, device=ids.device)
        )

        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, input_ids, max_new_tokens=50):
        # ids: (batch_size, seq_len)
        for _ in range(max_new_tokens):
            logits, _ = self(input_ids)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            """            
            Sample from the distribution:
            Instead of taking the argmax, we use multinomial sampling to allow for more diverse outputs.
            This is particularly useful in generation tasks where we want to avoid deterministic outputs.
            """
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids