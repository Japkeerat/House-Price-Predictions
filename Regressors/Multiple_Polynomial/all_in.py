"""
This class is responsible for making a polynomial Regressor that is
implemented with all-in approach. Meaning, all the dataset is passed for building
the model.
"""

from sklearn.preprocessing import PolynomialFeatures

def simple_multiple_pr(X_train, X_test, y_train, y_test):
    poly = PolynomialFeatures(degree = 2)
    X_train = poly.fit_transform(X_train)
    X_test = poly.transform(X_test)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_predict = regressor.predict(X_test)
    return y_predict