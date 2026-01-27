import numpy as np


class KNNRegression:
    def __init__(self, k=3, weighted=False):
        self.k = k
        self.weighted = weighted
        self.X_train = None
        self.y_train = None

    # only for evaluation NOT training
    @staticmethod
    def MSE(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X, y):
        # X: shape (n_samples, n_features)
        # y: shape (n_samples,)
        X = np.asarray(X)
        y = np.asarray(y)

        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # X: shape (m_samples, n_features)
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fitted before predicting.")

        # distances to all training points
        diff = X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)  # shape (m, n_train)

        # indices of the k nearest neighbors
        nn_indices = np.argsort(dists, axis=1)[:, : self.k]
        nn_dists = np.take_along_axis(dists, nn_indices, axis=1)
        nn_targets = self.y_train[nn_indices]

        if self.weighted:
            # nn_dists could be 0 so we avoid an error by adding 1e-8 to it
            weights = 1 / (nn_dists + 1e-8)
            weighted_sum = np.sum(weights * nn_targets, axis=1)
            weights_total = np.sum(weights, axis=1)
            return weighted_sum / weights_total
        else:
            return np.mean(nn_targets, axis=1)
