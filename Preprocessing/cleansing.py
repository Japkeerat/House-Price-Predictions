"""
This class is responsible for 
1. handling missing values in modified dataframe's feature set.
2. Feature scalling.
Feature scalling didn't put much of an impact on increasing accuracy, therefore,
it is not used anymore. Infact after removing feature scalling, models improved
by 5%.
"""

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

# Uses mean strategy to modify feature set and explains not to create any copy
def cleanse_features(feature_set):
    imp = Imputer(missing_values = "NaN", strategy = "mean", axis = 0, copy = False)
    imp.fit(feature_set[:,:])
    feature_set[:,:] = imp.transform(feature_set[:,:])
    # feature_set = scale_features(feature_set)
    return feature_set

# Responsible for feature scalling. Brings everything to the range [-1,1]
def scale_features(feature_set):
    scale = StandardScaler()
    feature_set = scale.fit_transform(feature_set)
    return feature_set