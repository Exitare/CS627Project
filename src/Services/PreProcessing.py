from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from Services import PreProcessing
from RuntimeContants import Runtime_File_Data

np.random.seed(10)


# https://scikit-learn.org/stable/modules/preprocessing.html
# https://chrisalbon.com/machine_learning/preprocessing_structured_data/convert_pandas_categorical_column_into_integers_for_scikit-learn/
# https://stackoverflow.com/questions/51741605/standardize-dataset-containing-too-large-values Scaler

def pre_process_data_set(df):
    """
    Prepare the dataset, by filling na, remove bad columns and convert factorial to numerical columns
    :param df:
    :return:
    """
    df.replace([np.inf, -np.inf], np.nan)
    df[df == np.inf] = np.nan
    df = PreProcessing.fill_na(df)
    df = PreProcessing.remove_bad_columns(df)
    df = PreProcessing.convert_factorial_to_numerical(df)
    Runtime_File_Data.EVALUATED_FILE_PREPROCESSED_DATA_SET = df


def variance_selection(X):
    """
    Transforms and selects features that are above a certain threshold
    :param X:
    :return:
    """
    try:
        selector = VarianceThreshold()
        X = selector.fit_transform(X)
        return X

    except ValueError:
        return 0


def normalize_X(X):
    """
    Standard Scaler to normalize the data using z-scores
    """
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    return X


def remove_bad_columns(df):
    """
    Removing columns which are not necessary for model training or predictions like job runner name or handler name
    :param df:
    :return:
    """
    columns = []
    if 'job_runner_name' in df.columns:
        columns.append('job_runner_name')

    if 'handler' in df.columns:
        columns.append('handler')

    if 'destination_id' in df.columns:
        columns.append('destination_id')

    if 'input_file' in df.columns:
        columns.append('input_file')

    for column in columns:
        del df[column]

    return df


def convert_factorial_to_numerical(df):
    """
    Converts categorical data columns to its numerical equivalent using scikits´ LabelEncoder
    :param df:
    :return:
    """
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


def remove_random_rows(df, amount):
    """

    """

    drop_indices = np.random.choice(len(df), amount, replace=False)

    if len(drop_indices) == 0:
        return df

    return np.delete(df, drop_indices)
