import pandas as pd
import os

# 设置数据路径
DATA_DIR = './data'
train_path = os.path.join(DATA_DIR, 'new_train.tsv')
test_path = os.path.join(DATA_DIR, 'new_test.tsv')

# 读取训练和测试数据（指定没有表头 + 自定义列名）
train_df = pd.read_csv(train_path, sep='\t', header=None, names=["text", "label"])
test_df = pd.read_csv(test_path, sep='\t', header=None, names=["text", "label"])

# 显示训练数据前几行
print("🔥 训练集前5行：")
print(train_df.head())

print("训练集所有列名：", train_df.columns)

# 显示训练集标签的分布
print("\n📊 标签分布情况：")
print(train_df["label"].value_counts())

# 显示测试集的样本数量
print(f"\n📦 测试集样本总数：{len(test_df)}")

from features import BoWVectorizer

# 将文本转为特征向量
texts = train_df['text'].tolist()
vectorizer = BoWVectorizer()
vectorizer.build_vocab(texts)
X_train = vectorizer.transform(texts)

print(f"\n🧮 训练集特征维度：{X_train.shape}")
