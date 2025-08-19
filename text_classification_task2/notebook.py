from collections import Counter

# 1. 模拟训练文本列表
texts = [
    "Kidman is really the only thing that's worth watching in Birthday Girl.",
    "Once you get into its rhythm ... the movie becomes a heady experience.",
    "I kept wishing I was watching a documentary about the wartime Navajos.",
    "Kinnear does n't aim for our sympathy.",
    "All ends well, sort of, but the frenzied comic moments never click."
]

# 2. 分词（简单空格分词）
tokenized_texts = [text.lower().split() for text in texts]

# 3. 统计词频
all_tokens = [token for text in tokenized_texts for token in text]
counter = Counter(all_tokens)

# 4. 建立词表
word2idx = {}
word2idx['<PAD>'] = 0
word2idx['<UNK>'] = 1

for idx, (word, freq) in enumerate(counter.most_common(), start=2):
    word2idx[word] = idx

# 5. 打印前10个词和对应的编号
for word in list(word2idx.keys())[:10]:
    print(word, word2idx[word])

def sentence_to_ids(sentence, word2idx, unk_id=1):
    """
    把一句话转成数字ID列表
    参数：
      sentence: 字符串，比如 "Kidman is really the only thing"
      word2idx: 词表字典，key是单词，value是ID
      unk_id: 遇到不在词表里的词用的ID，默认1（你可以调整）
    返回：
      id_list: 数字ID列表，比如 [5, 6, 7, 1, 8, 9]
    """
    tokens = sentence.lower().split()  # 简单分词
    id_list = [word2idx.get(token, unk_id) for token in tokens]
    return id_list

# 测试示例
example_sentence = "Kidman is really the only thing that isn't known"
print(sentence_to_ids(example_sentence, word2idx))
