from iris import iris_monkey
from boston import boston_monkey
from breastca import breast_monkey
import numpy as np

if "__main__" == __name__:
    iris = iris_monkey()
    iris.summary()
    iris.train()
    iris.test()
    #passing 1d ndarray is deprecated
    print(iris.predict(np.array([[1,2,3,4],[5, 2.9, 1, 0.2]])))
    boston = boston_monkey()
    boston.linearRegression()
    boston.ridgeRegression()
    #
    breast = breast_monkey()
    breast.summary()
    breast.trainAndScore()
    breast.trainAndScore(100)
    #testing score will be higher than training set if C is very small
    breast.trainAndScore(0.001)

