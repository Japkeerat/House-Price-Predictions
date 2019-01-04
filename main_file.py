import pandas as pd
import numpy as np
from Preprocessing import dataframe as df # Custom class
from Preprocessing import cleansing as cl # Custom class
from sklearn.model_selection import train_test_split
from Regressors.Random_Forest.all_in import simple_rfr # Custom class
from Regressors.Random_Forest.backward_elimination import be_rfr # Custom class
from Regressors.Multiple_Linear.all_in import simple_multiple_lr # Custom class
from Regressors.Multiple_Linear.backward_elimination import be_multiple_lr # Custom class
from Postprocessing.Rsquare import adj_R_square # Custom class
from Postprocessing.accuracy_of_model import get_accuracy

# Importing and reading the csv file for training model.
# Also dropping the column that contains ID of every row which is practically not useful.
training_file = "train.csv"
housing_price = pd.read_csv(training_file)
housing_price = housing_price.drop(['Id'], axis = 1)

# Handling NaN for columns that contain missing data
housing_price = df.modify_dataframe(housing_price)

# Selecting features and target from dataframe
features = housing_price.iloc[:,:-1].values
target = housing_price.iloc[:,-1:].values
target = np.squeeze(np.asarray(target))

# Handling all the missing data for floating point and integer datatype columns.
# Also does the feature scalling.
features = cl.cleanse_features(features)

# Splitting to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2)

# Running the algorithm for Random Forest with all in approach
y_predict = simple_rfr(X_train, X_test, y_train, y_test)
print(adj_R_square(y_test, y_predict, 79, 1460)) # 0.84944
print(get_accuracy(y_test, y_predict, 10)) # 65.4%

# Running the algorithm for Mutliple Linear Regression with all in approach
y_predict = simple_multiple_lr(X_train, X_test, y_train, y_test)
print(adj_R_square(y_test, y_predict, 79, 1460))
print(get_accuracy(y_test, y_predict, 10))

# Running the algorithm for Linear Regression with backward elimination approach
y_test_pred = be_multiple_lr(features, target)
y_test, y_predict = zip(*y_test_pred)
print(adj_R_square(y_test, y_predict, 79, 1460))
print(get_accuracy(y_test, y_predict, 10))

# Running the algorithm for Random Forest Regression with backward elimination approach
y_test_pred = be_rfr(features, target)
y_test, y_predict = zip(*y_test_pred)
print(adj_R_square(y_test, y_predict, 79, 1460))
print(get_accuracy(y_test, y_predict, 10))