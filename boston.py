from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

class boston_monkey:
    def __init__(self):
        self.boston = load_boston()
        self.X = PolynomialFeatures(degree = 2, include_bias= False).fit_transform(MinMaxScaler().fit_transform(self.boston.data))
        print("Boston data shape: {}".format(self.X.shape))

    def linearRegression(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.boston.target, random_state = 0)
        lr = LinearRegression().fit(self.X_train, self.y_train)
        print("Linear Regression Coefficients ({0}): {1}, Intercept: {2}".format(lr.coef_.shape, lr.coef_, lr.intercept_))

    def ridgeRegression(self):
        param_grid = {'alpha': [0.01, 0.1, 0.8, 1, 5, 10]}
        # It's actually wrong to apply GridSearchCV here since cross validation examines the performance against training data,
        # but alpha is designed to get better score during testing/predict by decreasing the performance on training data.
        # Therefore, the best_params_ from GridSearchCV can't be considered as the optimized param
        grid = GridSearchCV(Ridge(), param_grid, cv = 5)
        grid.fit(self.X, self.boston.target)
        print("Ridge Regression best score: {:.2f}".format(grid.best_score_))
        print("Ridge Regression best parameters: {}".format(grid.best_params_))

        r = Ridge(alpha = grid.best_params_['alpha']).fit(self.X_train, self.y_train)
        print("Ridge Regression score on test data using {0} is {1:.2f}".format(grid.best_params_, r.score(self.X_test, self.y_test)))

        r = Ridge(alpha = 0.1).fit(self.X_train, self.y_train)
        print("Ridge Regression score on test data using {{'alpha': 0.1}} is {:.2f}".format(r.score(self.X_test, self.y_test)))