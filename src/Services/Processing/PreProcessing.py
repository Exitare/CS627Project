from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import ast
import logging
from Services.Configuration.Config import Config

np.random.seed(10)


# https://scikit-learn.org/stable/modules/preprocessing.html
# https://chrisalbon.com/machine_learning/preprocessing_structured_data/convert_pandas_categorical_column_into_integers_for_scikit-learn/
# https://stackoverflow.com/questions/51741605/standardize-dataset-containing-too-large-values Scaler

def pre_process_data_set(df):
    """
    Prepare the data set, by filling na, remove bad columns and convert factorial to numerical columns
    :param df:
    :return:
    """
    df = remove_bad_columns(df)
    df.replace([np.inf, -np.inf], np.nan)
    df[df == np.inf] = np.nan
    df = fill_na(df)
    df = convert_factorial_to_numerical(df)
    # Remove columns only containing 0
    df = df[(df.T != 0).any()]
    return df


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

    bad_columns = []
    search_columns = [
        'job_runner_name',
        'handler',
        'destination_id',
        'input_file',
        'chromInfo',
        'workflow_invocation_uuid',
        'regionsFiles',
        'values',
        'regionsFile',
        '|__identifier__',
        'blackListFile',
    ]

    bad_ends = ['id', 'identifier', '__identifier__', 'indeces']

    bad_starts = ["__workflow_invocation_uuid__", "chromInfo", '__job_resource',
                  'reference_source', 'reference_genome', 'rg',
                  'readGroup', 'refGenomeSource', 'genomeSource'
                  ]

    for column in df.columns:
        for search in search_columns:
            if search in column:
                bad_columns.append(column)

    for column in df.columns:
        for word in bad_ends:
            if column.endswith(word):
                bad_columns.append(column)

        for word in bad_starts:
            if column.startswith(word):
                bad_columns.append(column)

    for column in df.columns:
        series = df[column].dropna()

        # trim string of ""   This is necessary to check if the parameter is full of list or dict objects
        if df[column].dtype == object and all(
                type(item) == str and item.startswith('"') and item.endswith('"') for item in series):
            try:
                df[column] = df[column].str[1:-1].astype(float)
            except:
                df[column] = df[column].str[1:-1]

        # if more than half of the rows have a unique value, remove the categorical feature
        if df[column].dtype == object and len(df[column].unique()) >= 0.5 * df.shape[0]:
            bad_columns.append(column)

        # if number empty is greater than half, remove
        if df[column].isnull().sum() >= 0.75 * df.shape[0]:
            bad_columns.append(column)

        # if the number of categories is greater than 10 remove
        if df[column].dtype == object and len(df[column].unique()) >= 100:
            bad_columns.append(column)

        # if the feature is a list remove
        if all(type(item) == str and item.startswith("[") and item.endswith("]") for item in
               series):  # and item.startswith("[{'src'")
            if all(type(ast.literal_eval(item)) == list for item in series):
                bad_columns.append(column)

        # if the feature is a dict remove
        if all(type(item) == str and item.startswith("{") and item.endswith("}") for item in
               series):  # and item.startswith("[{'src'")
            if all(type(ast.literal_eval(item)) == list for item in series):
                bad_columns.append(column)

    for column in df.columns:
        for bad in bad_columns:
            if bad in column and column in df.columns:
                if Config.DEBUG:
                    logging.debug(f"Throwing {column} away!")
                del df[bad]

    return df


def clean_data(df):
    # make an alert if the number of categories in a column exceeds threshold_num_of_categories
    threshold_num_of_categories = 30

    for column in df:
        try:
            df[column] = df[column].astype(float)
            df[column] = df[column].fillna(0)
        except (ValueError, TypeError) as ex:
            df[column] = df[column].astype(str)
            if len(df[column].unique()) > threshold_num_of_categories:
                print(f"Deleting column {column}, because too many categories.")
                del df[column]
                continue

        if df[column].is_monotonic:
            print(f"Deleting column {column}, because it is monotonic")
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
            df[column].fillna('True', inplace=True)

        elif False in df[column]:
            df[column].fillna('False', inplace=True)

        else:
            df[column].fillna('0', inplace=True)

    return df


def remove_random_rows(df, amount):
    """
    Remove random rows
    """

    drop_indices = np.random.choice(len(df), amount, replace=False)

    if len(drop_indices) == 0:
        return df

    return np.delete(df, drop_indices)
