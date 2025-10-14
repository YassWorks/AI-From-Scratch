import numpy as np


class SKSM:

    def __init__(self, d: int, vocabulary: list[str], learning_rate: float):
        self.d = d
        self.vocabulary = vocabulary
        self.voc_size = len(vocabulary)
        self.lr = learning_rate

        self.V = np.random.rand(self.d, self.voc_size)
        self.U = np.random.rand(self.d, self.voc_size)

    @staticmethod
    def softmax(vector):
        e = np.exp(vector - np.max(vector))
        return e / np.sum(e)

    def forward(self, vc):
        return SKSM.softmax(self.U.T @ vc)
    
    @staticmethod
    def preprocess(text: str):
        ...

    def train(self, corpus: str, m: int, epochs: int):
        
        L = len(corpus)
        corpus_list = SKSM.preprocess(corpus).split(" ")
        
        # using SGD
        for _ in range(epochs):
            
            dJ_dV = np.zeros_like(self.V)
            dJ_dU = np.zeros_like(self.U)
            
            for c in corpus_list:
                c_idx = self.vocabulary.index(c)
                window = corpus_list[max(0, c_idx - m): min(c_idx + m + 1, len(corpus_list))]
                context_idx = [self.vocabulary.index(w) for w in window if w != c]

                if not context_idx:
                    continue

                vc = self.V[:, c_idx:c_idx+1]
                p = self.forward(vc)

                # one-hot matrix for context words
                y = np.zeros((self.voc_size, len(context_idx)))
                y[context_idx, np.arange(len(context_idx))] = 1

                dvc = self.U @ (p - y)
                dJ_dV[:, c_idx:c_idx+1] += dvc.sum(axis=1, keepdims=True)

                dJ_dU += vc @ (p - y).T

            self.V = self.V - self.lr * dJ_dV
            self.U = self.U - self.lr * dJ_dU
