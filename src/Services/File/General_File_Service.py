from pathlib import Path
import shutil
import datetime
import os
import ntpath
import sys
from Services import PreProcessing
from RuntimeContants import Runtime_Folders, Runtime_Datasets
from Services.Config import Config
import pandas as pd


def create_tool_folder(filename: str):
    """
    Creates a folder containing all information about the tool, like runtime analysis, plots etc.
    :param filename:
    :return:
    """

    path = Path(f"{Runtime_Folders.CURRENT_WORKING_DIRECTORY}/{filename}")
    try:
        Path(path).mkdir(parents=True, exist_ok=True)

    except OSError as ex:
        print("Creation of tool directory %s failed" % path)
        print("Stopping application")
        remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
        print(ex)
        sys.exit()
    else:
        Runtime_Folders.CURRENT_EVALUATED_TOOL_DIRECTORY = path
        return path


def get_file_name(path):
    """
    Returns the filename
    :param path:
    :return:
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def create_evaluation_folder():
    """
    Creates the evaluation folder aka the root folder for each run of the application
    :return:
    """
    now = datetime.datetime.now()
    path = Path(f"{Config.DATA_RESULTS_DIRECTORY}/{now.strftime('%Y-%m-%d-%H-%M-%S')}")
    try:
        Path(path).mkdir(parents=True, exist_ok=True)

    except OSError as ex:
        print("Creation of the evaluation directory %s failed" % path)
        print("Stopping application")
        print(ex)
        sys.exit()
    else:
        Runtime_Folders.CURRENT_WORKING_DIRECTORY = path


def remove_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as ex:
        print(f"Could not delete folder {path}")
        print(ex)


def create_directory(path):
    """
    Creates a folder with the given path
    :param path:
    :return:
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)

    except OSError as ex:
        print(ex)
        print("Creation of the directory %s failed" % path)
        print("Stopping application")
        sys.exit()


def check_folder_integrity():
    """
    Checks if the Data folder structure is up to date given the options from the config
    :return:
    """

    print("Checking data folder integrity...")
    if not os.path.isdir(f"{Config.DATA_ROOT_DIRECTORY}"):
        print("Raw data directory not found. Creating...")
        create_directory(Config.DATA_ROOT_DIRECTORY)

    if not os.path.isdir(f"{Config.DATA_RAW_DIRECTORY}"):
        print("Raw data directory not found. Creating...")
        create_directory(Config.DATA_RAW_DIRECTORY)

    if not os.path.isdir(f"{Config.DATA_RESULTS_DIRECTORY}"):
        print("Results data directory not found. Creating...")
        create_directory(Config.DATA_RESULTS_DIRECTORY)

    print("Data folder checked and ready!")


def read_files(path: str):
    """
    Read all files and returns everything as list of data frames
    :param path:
    :return:
    """
    try:
        directory = os.fsencode(path)
        data_frames = dict()
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".csv") or filename.endswith(".tsv"):
                df = pd.read_csv(f"{Config.DATA_RAW_DIRECTORY}/{filename}")
                df = PreProcessing.fill_na(df)
                df = PreProcessing.remove_bad_columns(df)
                df = PreProcessing.convert_factorial_to_numerical(df)
                data_frames[filename] = df
                continue
            else:
                continue

        Runtime_Datasets.RAW_FILE_DATA_SETS = data_frames
    except OSError as ex:
        print(ex)
        remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
        sys.exit()


def create_csv_file(df, folder, name):
    """
    Writes a df to a given folder with the given name
    :param df:
    :param folder:
    :param name:
    :return:
    """
    if folder != "":
        path = os.path.join(folder, f"{name}.csv")
        df.to_csv(path, index=True)
