import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 中文字体路径，需根据你的系统路径调整
font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'  
myfont = FontProperties(fname=font_path)

def train_and_evaluate(ngram_range=(1,1)):
    print(f"\n=== 使用 ngram_range={ngram_range} 训练模型 ===")

    train_df = pd.read_csv('./data/new_train.tsv', sep='\t', header=None, names=['text', 'label'])
    test_df = pd.read_csv('./data/new_test.tsv', sep='\t', header=None, names=['text', 'label'])

    vectorizer = CountVectorizer(max_features=10000, ngram_range=ngram_range)
    X_train = vectorizer.fit_transform(train_df['text'])
    X_test = vectorizer.transform(test_df['text'])

    y_train = torch.tensor(train_df['label'].values, dtype=torch.long)
    y_test = torch.tensor(test_df['label'].values, dtype=torch.long)

    X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    input_dim = X_train.shape[1]
    num_classes = 5
    W = torch.randn(input_dim, num_classes, dtype=torch.float32) * 0.01
    b = torch.zeros(num_classes, dtype=torch.float32)
    W.requires_grad_()
    b.requires_grad_()

    lr = 0.1
    batch_size = 64
    epochs = 20

    def accuracy(preds, labels):
        pred_labels = preds.argmax(dim=1)
        return (pred_labels == labels).float().mean().item()

    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        perm = torch.randperm(X_train.size(0))
        total_loss = 0
        total_acc = 0

        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i:i+batch_size]
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            logits = X_batch @ W + b
            loss = F.cross_entropy(logits, y_batch)

            loss.backward()

            with torch.no_grad():
                W -= lr * W.grad
                b -= lr * b.grad
                W.grad.zero_()
                b.grad.zero_()

            total_loss += loss.item() * X_batch.size(0)
            total_acc += accuracy(logits, y_batch) * X_batch.size(0)

        avg_loss = total_loss / X_train.size(0)
        avg_acc = total_acc / X_train.size(0)

        with torch.no_grad():
            val_logits = X_val @ W + b
            val_loss = F.cross_entropy(val_logits, y_val).item()
            val_acc = accuracy(val_logits, y_val)

        train_acc_history.append(avg_acc)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} 训练loss: {avg_loss:.4f} 准确率: {avg_acc:.4f} 验证loss: {val_loss:.4f} 验证准确率: {val_acc:.4f}")

    with torch.no_grad():
        test_logits = X_test @ W + b
        test_preds = test_logits.argmax(dim=1)
        test_acc = accuracy(test_logits, y_test)

    print(f"测试集准确率: {test_acc:.4f}")

    output_path = f'./data/test_predictions_ngram_{ngram_range[0]}_{ngram_range[1]}.txt'
    np.savetxt(output_path, test_preds.numpy(), fmt='%d')
    print(f"测试集预测结果已保存到 {output_path}")

    return train_acc_history, val_acc_history

def plot_accuracy(train_acc_11, val_acc_11, train_acc_12, val_acc_12):
    epochs = range(1, len(train_acc_11) + 1)
    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_acc_11, 'o-', label='训练准确率 (ngram=1,1)', linewidth=2)
    plt.plot(epochs, val_acc_11, 'o--', label='验证准确率 (ngram=1,1)', linewidth=2)
    plt.plot(epochs, train_acc_12, 's-', label='训练准确率 (ngram=1,2)', linewidth=2)
    plt.plot(epochs, val_acc_12, 's--', label='验证准确率 (ngram=1,2)', linewidth=2)

    plt.title('训练与验证准确率变化曲线', fontproperties=myfont, fontsize=16)
    plt.xlabel('训练轮数 (epoch)', fontproperties=myfont, fontsize=14)
    plt.ylabel('准确率', fontproperties=myfont, fontsize=14)
    plt.legend(prop=myfont, fontsize=12)
    plt.grid(True)

    plt.savefig('accuracy_comparison.png')
    print("图片已保存：accuracy_comparison.png")
    plt.show()

if __name__ == '__main__':
    train_acc_11, val_acc_11 = train_and_evaluate(ngram_range=(1,1))
    train_acc_12, val_acc_12 = train_and_evaluate(ngram_range=(1,2))

    plot_accuracy(train_acc_11, val_acc_11, train_acc_12, val_acc_12)
