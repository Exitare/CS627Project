from sklearn import preprocessing
import numpy as np


# https://scikit-learn.org/stable/modules/preprocessing.html
# https://chrisalbon.com/machine_learning/preprocessing_structured_data/convert_pandas_categorical_column_into_integers_for_scikit-learn/

def normalize_X(X):
    return preprocessing.MinMaxScaler(X)


def remove_bad_columns(df):
    """
    :param df:
    :return:
    """

    print("Removing columns...")
    columns = []
    if 'job_runner_name' in df.columns:
        columns.append('job_runner_name')

    if 'handler' in df.columns:
        columns.append('handler')

    if 'destination_id' in df.columns:
        columns.append('destination_id')

    for column in columns:
        del df[column]

    return df


def convert_factorial_to_numerical(df):
    """
    Converts categorical data columns to its numerical equivalent using scikits´ LabelEncoder
    :param df:
    :return:
    """
    print("Decoding categorical data columns...")
    columns = df.select_dtypes(exclude=['int', 'float']).columns
    print(columns)
    le = preprocessing.LabelEncoder()
    print('here')
    for column in columns:
        print(column)
        le.fit(df[column])
        # le.fit_transform(df[column].astype(str))
        df[column] = le.transform(df[column])

    return df


def fill_na(df):
    numeric_columns = df.select_dtypes(exclude=['object']).columns
    categorical_columns = df.select_dtypes(exclude=['int', 'float']).columns

    for column in numeric_columns:
        df[column].fillna(0, inplace=True)

    for column in categorical_columns:
        df[column].fillna('0', inplace=True)

    return df
