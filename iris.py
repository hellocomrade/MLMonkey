"""
Iris Plants Database
====================

Notes
-----
Data Set Characteristics:
    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
    :Summary Statistics:

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20  0.76     0.9565  (high!)
    ============== ==== ==== ======= ===== ====================

    :Missing Attribute Values: None
    :Class Distribution: 33.3% for each of 3 classes.
    :Creator: R.A. Fisher
    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
    :Date: July, 1988

This is a copy of UCI ML iris datasets.
http://archive.ics.uci.edu/ml/datasets/Iris

The famous Iris database, first used by Sir R.A Fisher

This is perhaps the best known database to be found in the
pattern recognition literature.  Fisher's paper is a classic in the field and
is referenced frequently to this day.  (See Duda & Hart, for example.)  The
data set contains 3 classes of 50 instances each, where each class refers to a
type of iris plant.  One class is linearly separable from the other 2; the
latter are NOT linearly separable from each other.
"""
import numpy as np
import scipy.stats
from sys import stderr
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class iris_monkey:
    def __init__(self):
        self.dataset = load_iris()
        self.array = np.array(self.dataset['data'])

    #http://stats.stackexchange.com/questions/57776/what-is-class-correlation
    #https://en.wikipedia.org/wiki/Intraclass_correlation
    #ICC describes how strongly units in the same group resemble each other
    def summary(self):
        print("{0} columns of data, total {1} rows".format(self.dataset['data'].shape[1], self.dataset['data'].shape[0]))
        for (col, ftrName) in enumerate(self.dataset['feature_names']):
            print("{0} Min: {1:.2f}\tMax: {2:.2f}\tMean: {3:.2f}\tSTD: {4:.2f}\tClass Correlation: {5:.4f}".format(ftrName, np.min(self.array[:, col]), np.max(self.array[:, col]), np.mean(self.array[:, col]), np.std(self.array[:, col], ddof = 0), scipy.stats.pearsonr(self.array[:, col], self.dataset['target'])[0]))

    def train(self):
        #Prefix X is a naming convention for two dimensional array/matrix using as data
        #Prefix y is a naming convention for labels
        #random_state is the pseudorandom number generator, we use 0 here as a fixed seed to make the outcome deterministic
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset['data'], self.dataset['target'], random_state = 0)
        self.classifier = KNeighborsClassifier(n_neighbors = 1)
        self.classifier.fit(self.X_train, self.y_train)

    def test(self, **kwargs):
        y_pred = self.classifier.predict(kwargs['X_test']) if "X_test" in kwargs else self.classifier.predict(self.X_test)
        #y_pred == self.y_test works here because both operators are numpy.array, which probably has a different == operator, which returns an array as well
        print("Test set score: {:.2f}/{:.2f}".format(np.mean(y_pred == self.y_test), self.classifier.score(self.X_test, self.y_test)))

    def predict(self, x_test):
        if type(x_test) is not np.ndarray:
            print("x_test should be {}".format(np.ndarray.__name__), file = stderr)
            raise TypeError("Invalid x_test type")
        elif 2 == x_test.ndim and x_test.shape[1] != self.dataset['data'].shape[1]:
            print("2d x_test.shape[1] != {}".format(self.dataset['data'].shape[1]), file = stderr)
            raise ValueError("Invalid x_test")
        elif 1 == x_test.ndim and x_test.shape[0] != self.dataset['data'].shape[1]:
            print("1d x_test.shape[0] != {}".format(self.dataset['data'].shape[1]), file=stderr)
            raise ValueError("Invalid x_test")
        else:
            return self.dataset['target_names'][self.classifier.predict(x_test)]
