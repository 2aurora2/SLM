import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, dataset, type, max_len=512):
        super(MyDataset, self).__init__()
        import tiktoken
        self.encoder = tiktoken.get_encoding("gpt2")
        self.max_len = max_len

        self.encoded_data = []
        self.eos_token = self.encoder.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )

        if dataset == 'wikitext':
            from datasets import load_dataset
            raw_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
            raw_data = [item['text'] for item in raw_dataset[type] if item['text'].strip()]
        
        full_encoded = []
        for text in raw_data:
            encoded_text = self.encode(text)
            full_encoded.extend(encoded_text)
        
        for i in range(0, len(full_encoded), self.max_len):
            """
            to include the next token for loss computation
            """
            chunk = full_encoded[i : i + self.max_len + 1]
            if len(chunk) < self.max_len + 1:
                chunk += self.eos_token * (self.max_len + 1 - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target = torch.tensor(chunk[1:], dtype=torch.long)
        return ids, target

    def encode(self, text):
        return self.encoder.encode(text)

    def decode(self, ids):
        return self.encoder.decode(ids)