from pathlib import Path
import ntpath
import os
import sys
from RuntimeContants import Runtime_Datasets
from Services.Configuration.Config import Config
import pandas as pd
from collections import defaultdict


def get_file_name(path):
    """
    Returns the filename
    :param path:
    :return:
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def read_file(path: str):
    """
    Reads the file located at the given path
    :param path:
    :return:
    """
    try:
        return pd.read_csv(f"{Config.DATA_RAW_DIRECTORY}/{path}")
    except OSError as ex:
        if Config.DEBUG:
            print(ex)
        return None


def create_csv_file(df, folder, name):
    """
    Writes a df to a given folder with the given name
    :param df:
    :param folder:
    :param name:
    :return:
    """
    if folder != "" and not df.empty:
        path = os.path.join(folder, f"{name}.csv")
        df.to_csv(path, index=True)


def get_tool_version_files():
    """
    Detects different versions of the same tool and stores the paths for further evaluation
    :return:
    """
    similar_files = defaultdict(list)
    for path in Runtime_Datasets.RAW_FILE_PATHS:
        filename = get_file_name(path)
        filename = filename.rsplit('_', 1)[0]
        similar_files[filename].append(path)

    Runtime_Datasets.RAW_FILE_PATHS = similar_files
