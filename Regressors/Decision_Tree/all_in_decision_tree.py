"""
This class is responsible for making a Decision Tree Regressor that is
implemented with all-in approach. Meaning, all the dataset is passed for building
the model.
"""

from sklearn.tree import DecisionTreeRegressor

def simple_dtr(X_train, X_test, y_train, y_test):
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)
    y_predict = regressor.predict(X_test)
    return y_predict