from iris import iris_monkey
import numpy as np

if "__main__" == __name__:
    iris = iris_monkey()
    iris.summary()
    iris.train()
    iris.test()
    #passing 1d ndarray is deprecated
    print(iris.predict(np.array([[1,2,3,4],[5, 2.9, 1, 0.2]])))