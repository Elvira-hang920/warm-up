import torch

def load_addition_dataset(train_path, val_path, max_len=7):
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
    idx2char = ['<PAD>', '<SOS>', '<EOS>']

    def tokenize(text):
        return list(text.strip())

    def build_vocab(lines):
        for line in lines:
            for char in tokenize(line):
                if char not in vocab:
                    vocab[char] = len(vocab)
                    idx2char.append(char)

    def encode(line):
        tokens = [vocab[c] for c in tokenize(line)]
        tokens = [vocab['<SOS>']] + tokens + [vocab['<EOS>']]
        if len(tokens) < max_len + 2:
            tokens += [vocab['<PAD>']] * (max_len + 2 - len(tokens))
        else:
            tokens = tokens[:max_len + 2]
        return tokens

    def load_data(path):
        inputs, outputs = [], []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                input_seq, output_seq = line.strip().split('\t')
                inputs.append(input_seq)
                outputs.append(output_seq)
        return inputs, outputs

    # 加载训练数据并构建词表
    train_inputs, train_outputs = load_data(train_path)
    build_vocab(train_inputs + train_outputs)

    # 加载验证数据
    val_inputs, val_outputs = load_data(val_path)

    # 编码为张量
    def encode_dataset(inputs, outputs):
        input_ids = torch.tensor([encode(s) for s in inputs], dtype=torch.long)
        output_ids = torch.tensor([encode(s) for s in outputs], dtype=torch.long)
        return input_ids, output_ids

    train_input_ids, train_output_ids = encode_dataset(train_inputs, train_outputs)
    val_input_ids, val_output_ids = encode_dataset(val_inputs, val_outputs)

    return train_input_ids, train_output_ids, val_input_ids, val_output_ids, vocab, idx2char


