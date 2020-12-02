import math
import numpy as np


def __index_marks(nrows, chunk_size):
    return range(chunk_size, math.ceil(nrows / chunk_size) * chunk_size, chunk_size)


def split_df(dfm, chunk_size):
    """
    Splits the df into chunks
    """
    indices = __index_marks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)


def get_label_data(df, label: str):
    """
    Returns the data of the data which
    """
    if "Label" in df:
        return df[df["Label"] == label].copy()
    else:
        return None
