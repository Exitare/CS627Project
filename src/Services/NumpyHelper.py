import numpy as np
from Services import File
from RuntimeContants import Runtime_Folders


# TODO: maybe use agg instead of this
def get_mean_per_column_per_df(df):
    """
    Returns the mean of every column of the given df
    :param df:
    :return:
    """
    try:
        mean = []
        for column in df:
            mean.append(df[column].mean())

        return np.asarray(mean)
    except:
        File.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)


def get_var_per_column_per_df(df):
    var = []
    for column in df:
        var.append(df[column].var())

    return np.asarray(var)


def replace_column_with_array(df, column_id, array):
    """
    Replaces the given row id with the given array
    :param df:
    :param column_id:
    :param array:
    :return:
    """
    try:
        array = np.asarray(array)
        df.iloc[column_id] = array
        return df

    except:
        File.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)


def df_only_nan(df):
    return df.isnull().values.all(axis=0).all()
