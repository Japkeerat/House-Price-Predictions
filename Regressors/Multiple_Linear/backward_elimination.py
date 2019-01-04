"""
Feature Selection using Backward Elimination property and hence building a 
model on the selected features.
"""

import statsmodels.formula.api as sm
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression

def be_multiple_lr(features, target):
    features = add_ones(features)
    features = find_optimal_features(features, target)
    X_train, X_test, y_train, y_test = tts(features, target, test_size = 0.2)
    y_predict = build_model(X_train, X_test, y_train)
    test_pred = zip(y_test, y_predict)
    return test_pred
    

# Adding an array of 1 as column matrix to the features matrix to avoid the
# limitation of backward elimination
def add_ones(features):
    features = np.append(arr = np.ones((len(features), 1), dtype = int), values = features, axis = 1)
    return features


# Finding the columns that affect the most on model. This is kind of preprocessing
# work for model building. It returns the list of indexes of columns that have
# the most affect which is determined by the p-value of the column.
def find_optimal_features(features, target):
    significance_level = 0.05
    size = len(features[0])
    for i in range(0, size):
        reg_ols = sm.OLS(endog = target, exog = features).fit()
        max_var = max(reg_ols.pvalues).astype(float)
        if(max_var>significance_level):
            for j in range(0, size - i):
                if(reg_ols.pvalues[j].astype(float) == max_var):
                    features = np.delete(features, j, 1)
    return features


# This builds the Linear Regression model and also predicts values for some
# section of features
def build_model(X_train, X_test, y_train):
    regressor = LinearRegression(normalize = True, n_jobs = -1)
    regressor.fit(X_train, y_train)
    y_predict = regressor.predict(X_test)
    return y_predict