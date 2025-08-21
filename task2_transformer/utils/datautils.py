import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re

# =====================
# 工具函数
# =====================
def clean_text(text):
    """简单清理文本"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.strip()

def build_vocab(texts, min_freq=2):
    """建立词表"""
    from collections import Counter
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode_text(text, vocab, max_len=100):
    """把句子编码为数字"""
    tokens = text.split()
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    if len(ids) < max_len:
        ids = ids + [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

# =====================
# Dataset 类
# =====================
class TextDataset(Dataset):
    def __init__(self, path, vocab=None, max_len=100, build=False):
        df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"])
        df["text"] = df["text"].apply(clean_text)
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()

        if build:
            self.vocab = build_vocab(self.texts)
        else:
            self.vocab = vocab

        self.max_len = max_len
        self.encoded = [encode_text(t, self.vocab, self.max_len) for t in self.texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded[idx]), torch.tensor(self.labels[idx])

def load_dataloaders(train_path, test_path, batch_size=32, max_len=100):
    train_dataset = TextDataset(train_path, build=True, max_len=max_len)
    test_dataset = TextDataset(test_path, vocab=train_dataset.vocab, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, train_dataset.vocab
