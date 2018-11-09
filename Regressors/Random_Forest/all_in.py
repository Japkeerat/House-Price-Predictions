"""
This class is responsible for making a Random Forest Regressor that is
implemented with all-in approach. Meaning, all the dataset is passed for building
the model.
"""

from sklearn.ensemble import RandomForestRegressor

def simple_rfr(X_train, X_test, y_train, y_test):
    regressor = RandomForestRegressor(n_estimators = 750, criterion = "mae", warm_start = True)
    regressor.fit(X_train, y_train)
    y_predict = regressor.predict(X_test)
    return y_predict