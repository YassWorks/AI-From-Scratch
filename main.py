from Supervised_Learning.LinearRegression import LinearRegression
from Supervised_Learning.UnivariateLinearRegression import UnivariateLinearRegression
from Unsupervised_Learning.KMeans import KMeans
from Tools import CrossValidation
import numpy as np

def main():
    # create dummy data
    np.random.seed(0)
    x = np.random.rand(1000, 18)
    w = np.random.randn(18, 1)
    b = np.random.randn(1000, 1)
    y = np.dot(x, w) + b

    model = LinearRegression()
    cv = CrossValidation(folds=5)

    # get the score of the model
    score = cv.get_score(model, x, y)
    print("-"*30)
    print("My score:", score)
    print("-"*30)
    
if __name__ == "__main__":
    main()