from sklearn import preprocessing


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

    columns = ['job_runner_name', 'handler', 'destination_id']

    for column in columns:
        del df[column]
    # use pd to remove
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
        df[column] = le.transform(df[column])

    return df
