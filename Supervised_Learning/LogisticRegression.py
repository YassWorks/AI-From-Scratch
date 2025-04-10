import numpy as np

class LogisticRegression:
    
    def __init__(self, random_state=0, w=None, b=None, lr=0.0000001):
        np.random.seed(random_state)
        self.w = w
        self.b = b
        self.lr = lr
        
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def cross_entropy(y, y_hat):
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    
    def train(self, x_train, y_train, max_loss=0.1, verbose=False, max_iters=10000):
        # we must determine w and b so that the loss function is minimized
        self.w = np.zeros((x_train.shape[1], 1))
        self.b = 0    
        prev_loss = float('inf')
        
        for _ in range(max_iters):
            y_hat = self.predict(x_train)
            loss = self.cross_entropy(y_train, y_hat)
            if loss == float('inf'):
                break
            if verbose:
                print('Loss:', loss)
            if abs(prev_loss - loss) <= max_loss:
                break
            prev_loss = loss
            self.gradient_descent(x_train, y_train, y_hat)
            
    def predict(self, x):
        return self.sigmoid(np.dot(x, self.w) + self.b)
    
    def gradient_descent(self, x_train, y_train, y_hat):
        # derivative of the loss function with respect to w and b
        m = x_train.shape[0]
        dw = -(1/m) * np.dot(x_train.T, (y_train - y_hat)) 
        db = -(1/m) * np.sum(y_train - y_hat) 
        
        # update w and b
        self.w -= self.lr * dw
        self.b -= self.lr * db