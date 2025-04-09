import numpy as np

class LinearRegression:

    def __init__(self, random_state=0, w=None, b=None, lr=1e-22):
        np.random.seed(random_state)
        self.w = w
        self.b = b
        self.lr = lr

    @staticmethod
    def MSE(y, y_hat):
        return np.mean((y.reshape(-1, 1) - y_hat) ** 2)

    def train(self, x_train, y_train, max_loss=0.1, verbose=False, max_iters=10000):
        # we must determine w and b so that the loss function is minimized
        # we'll use the gradient descent algorithm to do this
        # initialize w and b with random values
        self.w = np.zeros((x_train.shape[1], 1))
        self.b = 0    
        prev_loss = float('inf')

        for _ in range(max_iters):
            y_hat = self.predict(x_train)
            loss = self.MSE(y_train, y_hat)
            if loss == float('inf'):
                break
            if verbose:
                print('Loss:', loss)
            if abs(prev_loss - loss) <= max_loss:  # stop if loss change is small enough
                break
            prev_loss = loss
            self.gradient_descent(x_train, y_train, y_hat)

    def predict(self, x):
        return np.dot(x, self.w) + self.b
    
    def gradient_descent(self, x_train, y_train, y_hat):
        # derivative of the loss function with respect to w and b
        m = x_train.shape[0]
        dw = -(2/m) * np.dot(x_train.T, (y_train - y_hat)) 
        db = -(2/m) * np.sum(y_train - y_hat) 
        
        # update w and b
        self.w -= self.lr * dw
        self.b -= self.lr * db