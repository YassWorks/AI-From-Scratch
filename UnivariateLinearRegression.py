

import numpy as np
import matplotlib.pyplot as plt

class UnivariateLinearRegression():

    def __init__(self, random_state=0, w=None, b=None, lr=0.1):
        np.random.seed(random_state)
        self.w = w
        self.b = b
        self.lr = lr

    @staticmethod
    def MSE(y, y_hat):
        return np.mean((y - y_hat) ** 2)

    def train(self, x_train, y_train, max_loss=0.1, verbose=False, max_iters=10000):
        # we must determine w and b so that the loss function is minimized
        # we'll use the gradient descent algorithm to do this

        # initialize w and b with random values
        self.w = np.random.randn()
        self.b = np.random.randn()
        
        prev_loss = float('inf')

        for _ in range(max_iters):
            y_hat = self.predict(x_train)
            loss = self.MSE(y_train, y_hat)
            if verbose:
                print('Loss:', loss)
            if abs(prev_loss - loss) < max_loss:  # Stop if loss change is small
                break
            prev_loss = loss
            self.gradient_descent(x_train, y_train, y_hat)


    def predict(self, x):
        return self.w * x + self.b
    
    def gradient_descent(self, x_train, y_train, y_hat):
        # derivative of the loss function with respect to w and b
        dL_dw = np.mean(x_train * (y_train - y_hat))
        dL_db = np.mean(y_train - y_hat)

        # update w and b
        self.w += self.lr * dL_dw
        self.b += self.lr * dL_db

    def plot(self, x_train, y_train):
        plt.plot(x_train, y_train, 'o')
        plt.plot(x_train, self.predict(x_train), 'r')
        plt.show()
