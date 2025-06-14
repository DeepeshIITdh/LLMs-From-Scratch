import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import re


class HPotterDataset(Dataset):
    def __init__(self, text, tokenizer, max_length=256, stride=128):
        self.tokenizer = tokenizer
        token_ids = tokenizer.encode_ordinary(text)
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i+max_length], dtype=torch.long))
            self.target_ids.append(torch.tensor(token_ids[i+1:i+max_length+1], dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def clean_text(text):
    """Clean Harry Potter specific artifacts"""
    text = re.sub(r'CHAPTER [IVXLCDM]+\n\n', '', text)  # Remove chapter numbers
    text = text.replace('“', '"').replace('”', '"')  # Standardize quotes
    text = re.sub(r'\n+', '\n', text)  # Normalize newlines
    return text.strip()

def load_data(batch_size=32):
    """Main data loading function"""
    with open('harrypotter.txt', 'r', encoding='utf-8') as f:
        text = clean_text(f.read())

    tokenizer = tiktoken.get_encoding("gpt2")
    train_size = int(0.9 * len(text))

    train_data = HPotterDataset(text[:train_size], tokenizer)
    val_data = HPotterDataset(text[train_size:], tokenizer)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=2)

    return train_loader, val_loader, tokenizer