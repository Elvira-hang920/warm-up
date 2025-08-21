import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re

# =====================
# 工具函数
# =====================
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

class TextDataset(Dataset):
    def __init__(self, texts, labels, vectorizer=None):
        self.texts = texts
        self.labels = labels
        if vectorizer is None:
            self.vectorizer = CountVectorizer(max_features=10000, ngram_range=(1,2))
            self.features = self.vectorizer.fit_transform(self.texts).toarray()
        else:
            self.vectorizer = vectorizer
            self.features = self.vectorizer.transform(self.texts).toarray()
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.vocab_size = self.features.shape[1]
        self.num_classes = len(np.unique(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_dataloaders(train_file, test_file, batch_size=32, test_size=0.2):
    train_df = pd.read_csv(train_file, sep='\t', header=None, names=['text', 'label'])
    test_df = pd.read_csv(test_file, sep='\t', header=None, names=['text', 'label'])

    train_texts, train_labels = train_df['text'].tolist(), train_df['label'].tolist()
    test_texts, test_labels = test_df['text'].tolist(), test_df['label'].tolist()

    train_dataset = TextDataset(train_texts, train_labels)
    test_dataset = TextDataset(test_texts, test_labels, vectorizer=train_dataset.vectorizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
