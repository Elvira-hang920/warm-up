import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from datautils import preprocess_dataset
from textcnn import TextCNN
import torch.optim as optim

def main():
    # 参数设置
    data_path = 'data/new_train.tsv'
    max_len = 50
    min_freq = 2
    batch_size = 64
    embed_dim = 100
    class_num = 5
    epochs = 10
    lr = 0.001
    valid_ratio = 0.1  # 验证集占10%
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 数据预处理
    print("预处理数据...")
    data, labels, word2idx = preprocess_dataset(data_path, max_len=max_len, min_freq=min_freq)
    data = torch.tensor(data, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    # 2. 划分训练集和验证集
    total_samples = len(data)
    valid_size = int(total_samples * valid_ratio)
    train_size = total_samples - valid_size
    train_data, valid_data = random_split(TensorDataset(data, labels), [train_size, valid_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    # 3. 初始化模型
    model = TextCNN(vocab_size=len(word2idx), embed_dim=embed_dim, class_num=class_num)
    model = model.to(device)

    # 4. 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-5)

    # 5. 训练与验证循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        valid_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - 训练损失: {avg_loss:.4f} - 验证准确率: {valid_acc:.4f}")

    # 6. 保存模型
    torch.save(model.state_dict(), 'model.pth')
    print("模型已保存到 model.pth")

if __name__ == '__main__':
    main()
