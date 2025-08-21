import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, num_classes, max_len=100, dropout=0.1):
        super().__init__()

        # Embedding + 位置编码
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        # Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=hidden_dim, dropout=dropout,
                                                   activation="relu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bsz, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)

        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.transformer(x)  # [batch, seq_len, embed_dim]

        x = x.mean(dim=1)  # 平均池化
        x = self.dropout(x)
        return self.fc(x)
