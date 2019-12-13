from sklearn import preprocessing
import numpy as np


# https://scikit-learn.org/stable/modules/preprocessing.html
# https://chrisalbon.com/machine_learning/preprocessing_structured_data/convert_pandas_categorical_column_into_integers_for_scikit-learn/

def normalize_X(X):
    return X
    # return preprocessing.scale(X)


def remove_bad_columns(df):
    """
    Removing columns which are not necessary for model training or predictions like job runner name or handler name
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
    le = preprocessing.LabelEncoder()
    for column in columns:
        le.fit(df[column])
        # le.fit_transform(df[column].astype(str))
        df[column] = le.transform(df[column])

    return df


def fill_na(df):
    """
    Filling all NAs.
    Changing boolean values to string values to replace them.
    """
    numeric_columns = df.select_dtypes(exclude=['object']).columns
    categorical_columns = df.select_dtypes(exclude=['int', 'float']).columns

    mask = df.applymap(type) != bool
    d = {True: 'True', False: 'False'}
    df = df.where(mask, df.replace(d))

    for column in numeric_columns:
        df[column].fillna(0, inplace=True)

    for column in categorical_columns:
        if True in df[column]:
            df[column].fillna('False', inplace=True)

        elif False in df[column]:
            df[column].fillna('False', inplace=True)

        else:
            df[column].fillna('0', inplace=True)

    return df
