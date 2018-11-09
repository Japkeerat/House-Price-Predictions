"""
This class builds a simple linear regression model that takes all column into
account.
"""

from sklearn.linear_model import LinearRegression

# Builds a linear regression model
def simple_multiple_lr(X_train, X_test, y_train, y_test):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_predict = regressor.predict(X_test)
    return y_predict