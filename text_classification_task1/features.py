import numpy as np

class BoWVectorizer:
    def __init__(self):
        self.word2id = {}

    def build_vocab(self, texts):
        idx = 0
        for text in texts:
            for word in text.split():
                if word not in self.word2id:
                    self.word2id[word] = idx
                    idx += 1

    def transform(self, texts):
        vectors = []
        for text in texts:
            vec = [0] * len(self.word2id)
            for word in text.split():
                if word in self.word2id:
                    vec[self.word2id[word]] += 1
            vectors.append(vec)
        return np.array(vectors)
