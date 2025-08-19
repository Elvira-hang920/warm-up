import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, class_num, kernel_nums=[100,100,100], kernel_sizes=[3,4,5], dropout=0.5, pretrained_embeddings=None, freeze_embed=False):
        super(TextCNN, self).__init__()

        # 1. Embedding 层
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embed, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 2. 卷积层，多个卷积核大小
        self.convs = nn.ModuleList([
            nn.Conv2d(1, kernel_nums[i], (kernel_sizes[i], embed_dim))
            for i in range(len(kernel_sizes))
        ])

        # 3. dropout 层
        self.dropout = nn.Dropout(dropout)

        # 4. 输出全连接层
        self.fc = nn.Linear(sum(kernel_nums), class_num)

    def forward(self, x):
        # x: [batch_size, max_len]
        x = self.embedding(x)  # [batch_size, max_len, embed_dim]

        x = x.unsqueeze(1)  # [batch_size, 1, max_len, embed_dim]

        # 卷积 + 激活 + 池化
        conv_results = []
        for conv in self.convs:
            c = conv(x)  # [batch_size, kernel_num, L_out, 1]
            c = F.relu(c.squeeze(3))  # [batch_size, kernel_num, L_out]
            c = F.max_pool1d(c, c.size(2)).squeeze(2)  # [batch_size, kernel_num]
            conv_results.append(c)

        out = torch.cat(conv_results, 1)  # [batch_size, sum(kernel_nums)]

        out = self.dropout(out)

        logits = self.fc(out)  # [batch_size, class_num]

        return logits
