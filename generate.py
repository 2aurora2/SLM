import torch
import tiktoken
from model import GPT
from train import GPTConfig

CKPT_PATH = "checkpoints/checkpoint_10.pt"  
PROMPT = "The future of AI is"             
MAX_NEW = 200                               
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------------

def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = GPT(GPTConfig())               
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model

def generate(model, prompt, max_new):
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    ids = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    max_seq = GPTConfig.max_seq
    for _ in range(max_new):
        if ids.size(1) >= max_seq:          
            ids = ids[:, -max_seq+1:]
        logits, _ = model(ids)
        nxt = torch.multinomial(
            torch.softmax(logits[:, -1, :], dim=-1), num_samples=1
        )
        ids = torch.cat([ids, nxt], dim=1)
    return enc.decode(ids[0].tolist())

if __name__ == "__main__":
    model = load_model(CKPT_PATH, DEVICE)
    print(generate(model, PROMPT, MAX_NEW))