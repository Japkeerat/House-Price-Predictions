"""
This class is responsible for handling those columns of dataframe that have a
datatype as object and contain missing data in them. Basically it would replace
all the missing data in those columns with NONE so as the Encoding can be done.
Then it encodes the categorical data to floating point numbers.
"""

# Makes a list of columns with datatype as object
def modify_dataframe(df):
    dataframe = df.copy()
    object_list = list(dataframe.select_dtypes(include = ['object']).columns)
    dataframe = handle_NaN_for_object(dataframe, object_list)
    dataframe = encode(dataframe, object_list)
    return dataframe

# Replaces NaN with None
def handle_NaN_for_object(dataframe, object_list):
    for column in object_list:
        dataframe[column] = dataframe[column].fillna('AAAA')
    return dataframe


def encode(dataframe, object_list):
    for column in object_list:
        dataframe[column] = dataframe[column].astype('category').cat.codes
    return dataframe