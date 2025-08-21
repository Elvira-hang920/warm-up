import torch
import torch.nn as nn
import torch.optim as optim
from preprocess_dataset import load_dataloaders
from transformer_model import TransformerClassifier

# =====================
# 参数设置
# =====================
train_path = "./data/new_train.tsv"
test_path = "./data/new_test.tsv"

batch_size = 64
max_len = 100
embed_dim = 128
num_heads = 4
num_layers = 2
hidden_dim = 256
num_classes = 5  # 标签范围 0~4
epochs = 5
lr = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# 数据加载
# =====================
train_loader, test_loader, vocab = load_dataloaders(train_path, test_path, batch_size=batch_size, max_len=max_len)
print(f"Vocab size: {len(vocab)}")

# =====================
# 模型定义
# =====================
model = TransformerClassifier(vocab_size=len(vocab),
                              embed_dim=embed_dim,
                              num_heads=num_heads,
                              num_layers=num_layers,
                              hidden_dim=hidden_dim,
                              num_classes=num_classes,
                              max_len=max_len).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# =====================
# 训练函数
# =====================
def train():
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# =====================
# 验证函数
# =====================
def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# =====================
# 主循环
# =====================
for epoch in range(1, epochs + 1):
    loss = train()
    acc = evaluate()
    print(f"Epoch {epoch}: Loss={loss:.4f}, Test Acc={acc:.4f}")
