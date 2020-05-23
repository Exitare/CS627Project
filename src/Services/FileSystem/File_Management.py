from pathlib import Path
import ntpath
import os
import sys
from RuntimeContants import Runtime_Folders, Runtime_Datasets, Runtime_File_Data
from Services.Config import Config
from Services.FileSystem import FolderManagement
import pandas as pd
from collections import defaultdict


def load_required_data():
    """
    Gathers all data, which is required. Takes command line args into account.
    :return:
    """
    if Runtime_Datasets.COMMAND_LINE_ARGS.merge:
        get_tool_version_files()

    get_all_file_paths(Config.DATA_RAW_DIRECTORY)


def get_file_name(path):
    """
    Returns the filename
    :param path:
    :return:
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_all_file_paths(path: str):
    """
    Iterates through the folder and store every file path.
    :param path:
    :return:
    """
    try:
        print("Loading files")
        directory = Path(path)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".csv") or filename.endswith(".tsv"):
                Runtime_Datasets.RAW_FILE_PATHS.append(filename)

    except OSError as ex:
        print(ex)
        FolderManagement.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
        sys.exit()


def read_file(path: str):
    """
    Reads the file located by the given path
    :param path:
    :return:
    """
    try:
        df = pd.read_csv(f"{Config.DATA_RAW_DIRECTORY}/{path}")
        Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET = df
    except OSError as ex:
        print(ex)
        FolderManagement.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
        sys.exit()


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
