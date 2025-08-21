import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ------------------ 动态添加路径 ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'models'))

from datautils import load_dataloaders  # utils 下
from transformer import TransformerModel  # models 下

# ------------------ 参数配置 ------------------
TRAIN_FILE = os.path.join(BASE_DIR, 'data', 'new_train.tsv')
TEST_FILE = os.path.join(BASE_DIR, 'data', 'new_test.tsv')
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "transformer_model.pt")

# ------------------ 数据加载 ------------------
train_loader, test_loader = load_dataloaders(TRAIN_FILE, TEST_FILE, batch_size=BATCH_SIZE)

# ------------------ 模型初始化 ------------------
vocab_size = train_loader.dataset.vocab_size if hasattr(train_loader.dataset, 'vocab_size') else 10000
num_classes = train_loader.dataset.num_classes if hasattr(train_loader.dataset, 'num_classes') else 5

model = TransformerModel(vocab_size=vocab_size, embed_dim=128, num_classes=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR,weight_decay=1e-5)

# ------------------ 训练循环 ------------------
train_losses = []
val_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    best_val = 0
    patience = 0
    max_patience = 3

    # ------------------ 验证 ------------------
    model.eval()
    correct = 0
    total = 0
    val_loss_total=0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss=criterion(outputs,labels)
            val_loss_total += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total
    val_accuracies.append(val_acc)
    val_loss_avg = val_loss_total/len(test_loader)

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    if val_acc > best_val:
        best_val = val_acc
        patience = 0
        torch.save(model.state_dict(),MODEL_SAVE_PATH)
    else:
        patience += 1
    if patience >= max_patience:
        print("Early stopping triggered")

# ------------------ 保存模型 ------------------
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# ------------------ 绘图 ------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(range(1, EPOCHS+1), train_losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(range(1, EPOCHS+1), val_accuracies, marker='o', color='orange')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "training_curves.png"))
plt.show()
