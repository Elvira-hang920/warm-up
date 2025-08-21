import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, num_classes=5, num_heads=4, num_layers=2, ff_hidden_dim=256, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  
        self.dropout=nn.Dropout(0.2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_hidden_dim, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, vocab_size]
        x = self.embedding(x)  # [batch_size, embed_dim]
        #x = x.unsqueeze(1)  # [batch_size, seq_len=1, embed_dim]
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch, embed_dim]
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # 平均池化得到 [batch_size, embed_dim]
        out = self.fc(x)
        return out

