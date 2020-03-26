import datetime
import os
import ntpath
import sys
from Services import PreProcessing, NumpyHelper
from Services.Config import Config
import pandas as pd
import Constants
from pathlib import Path
import shutil


def write_too_small_data_sets():
    """
    Creates a csv containing all files which does not have enough data
    to generate reasonable predictions
    """
    df = pd.DataFrame(Constants.FILES_CONTAINING_NOT_ENOUGH_DATA, columns = "Filename")
    df.to_csv(f"{Constants.CURRENT_WORKING_DIRECTORY}/Small_Data_Sets.csv")


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
        Path(path).mkdir(parents=True, exist_ok=True)

    except OSError as ex:
        print("Creation of tool directory %s failed" % path)
        print("Stopping application")
        remove_folder(Constants.CURRENT_WORKING_DIRECTORY)
        print(ex)
        sys.exit()
    else:
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
        Path(path).mkdir(parents=True, exist_ok=True)

    except OSError as ex:
        print("Creation of the evaluation directory %s failed" % path)
        print("Stopping application")
        print(ex)
        sys.exit()
    else:
        Constants.CURRENT_WORKING_DIRECTORY = path


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
                df = PreProcessing.fill_na(df)
                df = PreProcessing.remove_bad_columns(df)
                df = PreProcessing.convert_factorial_to_numerical(df)
                data_frames[filename] = df
                continue
            else:
                continue

        return data_frames
    except OSError as ex:
        print(ex)
        remove_folder(Constants.CURRENT_WORKING_DIRECTORY)
        sys.exit()


def write_summary():
    if not NumpyHelper.df_only_nan(Constants.RUNTIME_MEAN_REPORT):
        Constants.RUNTIME_MEAN_REPORT['file'] = Constants.EVALUATED_FILE_NAMES
        Constants.RUNTIME_MEAN_REPORT['row_count'] = Constants.EVALUATED_FILE_ROW_COUNTS
        Constants.RUNTIME_MEAN_REPORT['parameter_count'] = Constants.EVALUATED_FILE_PARAMETER_COUNTS
        create_csv_file(Constants.RUNTIME_MEAN_REPORT, Constants.CURRENT_WORKING_DIRECTORY,
                        Config.FILE_RUNTIME_MEAN_SUMMARY)

    if not NumpyHelper.df_only_nan(Constants.RUNTIME_VAR_REPORT):
        Constants.RUNTIME_VAR_REPORT['file'] = Constants.EVALUATED_FILE_NAMES
        Constants.RUNTIME_VAR_REPORT['row_count'] = Constants.EVALUATED_FILE_ROW_COUNTS
        Constants.RUNTIME_VAR_REPORT['parameter_count'] = Constants.EVALUATED_FILE_PARAMETER_COUNTS
        create_csv_file(Constants.RUNTIME_VAR_REPORT, Constants.CURRENT_WORKING_DIRECTORY,
                        Config.FILE_RUNTIME_VAR_SUMMARY)

    if not NumpyHelper.df_only_nan(Constants.MEMORY_MEAN_REPORT):
        Constants.MEMORY_MEAN_REPORT['file'] = Constants.EVALUATED_FILE_NAMES
        Constants.MEMORY_MEAN_REPORT['row_count'] = Constants.EVALUATED_FILE_ROW_COUNTS
        Constants.MEMORY_MEAN_REPORT['parameter_count'] = Constants.EVALUATED_FILE_PARAMETER_COUNTS
        create_csv_file(Constants.MEMORY_MEAN_REPORT, Constants.CURRENT_WORKING_DIRECTORY,
                        Config.FILE_MEMORY_MEAN_SUMMARY)

    if not NumpyHelper.df_only_nan(Constants.MEMORY_VAR_REPORT):
        Constants.MEMORY_VAR_REPORT['file'] = Constants.EVALUATED_FILE_NAMES
        Constants.MEMORY_VAR_REPORT['row_count'] = Constants.EVALUATED_FILE_ROW_COUNTS
        Constants.MEMORY_VAR_REPORT['parameter_count'] = Constants.EVALUATED_FILE_PARAMETER_COUNTS
        create_csv_file(Constants.MEMORY_VAR_REPORT, Constants.CURRENT_WORKING_DIRECTORY,
                        Config.FILE_MEMORY_VAR_SUMMARY)
