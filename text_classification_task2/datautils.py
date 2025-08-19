from collections import Counter

def load_tsv(path):
    texts = []
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            splits = line.split('\t')
            if len(splits) != 2:
                continue
            text, label = splits
            texts.append(text)
            labels.append(int(label))
    return texts, labels

def build_word_vocab(texts, min_freq=1):
    counter = Counter()
    for sent in texts:
        tokens = sent.lower().split()
        counter.update(tokens)

    word2idx = {'<PAD>':0, '<UNK>':1}
    idx = 2

    for word, freq in counter.items():
        if freq >= min_freq:
            word2idx[word] = idx
            idx += 1
    return word2idx

def sentence_to_ids(sentence, word2idx, unk_id=1):
    tokens = sentence.lower().split()
    return [word2idx.get(token, unk_id) for token in tokens]

def pad_sequence(id_list, max_len, pad_id=0):
    if len(id_list) >= max_len:
        return id_list[:max_len]
    else:
        return id_list + [pad_id] * (max_len - len(id_list))

def preprocess_dataset(path, max_len=50, min_freq=1):
    # 1. 读取数据
    texts, labels = load_tsv(path)

    # 2. 建词表（只针对训练集用min_freq过滤）
    word2idx = build_word_vocab(texts, min_freq=min_freq)

    # 3. 文本转id + padding
    processed_texts = []
    for sent in texts:
        ids = sentence_to_ids(sent, word2idx)
        padded_ids = pad_sequence(ids, max_len)
        processed_texts.append(padded_ids)

    return processed_texts, labels, word2idx
