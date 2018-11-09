"""
This class is responsible for making a Support Vector Regressor that is
implemented with all-in approach. Meaning, all the dataset is passed for building
the model.
"""

from sklearn.svm import SVR

def simple_svr(X_train, X_test, y_train, y_test):
    regressor = SVR(kernel = 'sigmoid')
    regressor.fit(X_train, y_train)
    y_predict = regressor.predict(X_test)
    return y_predict