import numpy as np


class CrossValidation:

    def __init__(self, folds=5, random_state=0):
        np.random.seed(random_state)
        self.folds = folds

    def get_score(self, model, x, y):
        # we'll get the avg score of all folds
        scores = []
        for i in range(self.folds):
            x_train, y_train, x_test, y_test = self.get_fold(x, y, i)
            model.train(x_train, y_train)
            scores.append(model.MSE(y_test, model.predict(x_test)))
        return np.mean(scores)

    def get_fold(self, x, y, i):
        # get the i-th fold
        n = x.shape[0]
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_size = n // self.folds
        test_indices = indices[i * fold_size : (i + 1) * fold_size]
        train_indices = np.concatenate(
            [indices[: i * fold_size], indices[(i + 1) * fold_size :]]
        )
        return x[train_indices], y[train_indices], x[test_indices], y[test_indices]
