import datetime
from pathlib import Path
from Services.Configuration.Config import Config
from RuntimeContants import Runtime_Folders
import sys
import shutil


def initialize():
    """
    Checks if all required folders are present and creates the new evaluation folder for the ongoing evaluation
    :return:
    """
    check_required_folder_integrity()
    create_evaluation_folder()


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
        return path


def check_required_folder_integrity():
    """
    Checks if the Data folder structure is up to date given the options from the config
    :return:
    """

    data_root = Path(Config.DATA_ROOT_DIRECTORY)
    data_raw = Path(Config.DATA_RAW_DIRECTORY)
    data_results = Path(Config.DATA_RESULTS_DIRECTORY)

    print("Checking data folder integrity...")
    if not data_root.is_dir():
        print("Raw data directory not found. Creating...")
        create_directory(Config.DATA_ROOT_DIRECTORY)

    if not data_raw.is_dir():
        print("Raw data directory not found. Creating...")
        create_directory(Config.DATA_RAW_DIRECTORY)

    if not data_results.is_dir():
        print("Results data directory not found. Creating...")
        create_directory(Config.DATA_RESULTS_DIRECTORY)

    print("Data folder checked and ready!")
