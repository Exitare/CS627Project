import datetime
import os
import ntpath
import sys
from Services.Config import Config
from Services.PreProcessing import convert_factorial_to_numerical, remove_bad_columns, fill_na
import pandas as pd
import Constants
from pathlib import Path


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


def create_tool_folder(filename: str):
    """
    Creates a folder containing all information about the tool, like runtime analysis, plots etc.
    :param filename:
    :return:
    """

    path = Path(f"{Constants.CURRENT_WORKING_DIRECTORY}/{filename}")
    try:
        os.mkdir(path)

    except OSError as ex:
        print("Creation of the tool directory %s failed" % path)
        print("Stopping application")
        remove_folder(Constants.CURRENT_WORKING_DIRECTORY)
        print(ex)
        sys.exit()
    else:
        print("Successfully created the directory %s " % path)
        Constants.CURRENT_EVALUATED_TOOL_DIRECTORY = path
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
        os.mkdir(path)

    except OSError as ex:
        print("Creation of the directory %s failed" % path)
        print("Stopping application")
        print(ex)
        sys.exit()
    else:
        Constants.CURRENT_WORKING_DIRECTORY = path


def remove_folder(path):
    try:
        os.rmdir(path)
    except OSError:
        print(f"Could not delete folder {path}")


def create_directory(path):
    """
    Creates a folder with the given path
    :param path:
    :return:
    """
    print(path)
    try:
        os.mkdir(path)

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
        create_directory(Config.DATA_ROOT_DIRECTORY)

    if not os.path.isdir(f"{Config.DATA_RESULTS_DIRECTORY}"):
        print("Results data directory not found. Creating...")
        create_directory(Config.DATA_RESULTS_DIRECTORY)

    print("Data folder checked and ready!")


def read_files(path: str):
    """
    Read all files and returns everything as list of dataframes
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
                df = fill_na(df)
                df = remove_bad_columns(df)
                df = convert_factorial_to_numerical(df)
                data_frames[filename] = df
                continue
            else:
                continue

        return data_frames
    except OSError as ex:
        print(ex)
        remove_folder(Constants.CURRENT_WORKING_DIRECTORY)
        sys.exit()
