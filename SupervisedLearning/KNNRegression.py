import numpy as np

class KNNRegression:
    
    def __init__(self, k=3):
        self.k = k
    
    @staticmethod
    def MSE(y, y_hat):
        return np.mean((y.reshape(-1, 1) - y_hat) ** 2)