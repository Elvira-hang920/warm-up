import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datautils_addition import load_addition_dataset
from transformer_addition import TransformerAdditionModel

def train():
    # 参数设置
    train_path = 'data/addition_train.tsv'
    val_path = 'data/addition_val.tsv'  # ✅ 新增验证集路径
    max_len = 7  # "999+999" 长度为 7
    batch_size = 64
    embed_dim = 128
    num_heads = 4
    num_layers = 2
    hidden_dim = 512
    epochs = 10
    lr = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据准备
    print("加载数据...")
    train_input, train_output, val_input, val_output, vocab, idx2char = load_addition_dataset(
        train_path, val_path, max_len=max_len
    )

    # 构建训练集和验证集的TensorDataset和DataLoader
    train_dataset = TensorDataset(train_input, train_output)
    val_dataset = TensorDataset(val_input, val_output)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 模型初始化
    model = TransformerAdditionModel(
        vocab_size=len(vocab),
        d_model=embed_dim,
        nhead=num_heads,
        num_layers=num_layers,
        dim_feedforward=hidden_dim
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)

            # 拆分 tgt: 用前面的 token 预测后面的 token
            tgt_input = tgt[:, :-1]           # decoder 输入
            tgt_output = tgt[:, 1:]           # 目标标签

            optimizer.zero_grad()
            output = model(src, tgt_input)    # output: [batch, seq_len - 1, vocab_size]

            loss = criterion(
                output.view(-1, output.size(-1)), 
                tgt_output.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                output = model(src, tgt_input)        # [batch, seq_len - 1, vocab_size]
                pred = output.argmax(dim=-1)          # [batch, seq_len - 1]

                correct += (pred == tgt_output).sum().item()
                total += tgt_output.numel()

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - 训练损失: {avg_loss:.4f} - 验证准确率: {val_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'transformer_addition.pth')
    print("模型已保存到 transformer_addition.pth")

if __name__ == "__main__":
    train()


