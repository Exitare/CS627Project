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
    path = Path(Config.DATA_RESULTS_DIRECTORY, now.strftime('%Y-%m-%d-%H-%M-%S'))
    try:
        Path(path).mkdir(parents=True, exist_ok=True)

    except OSError as ex:
        print(f"Could not create evaluation directory {path}")
        print("Stopping application")
        print(ex)
        sys.exit()
    else:
        Runtime_Folders.EVALUATION_DIRECTORY = path


def remove_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as ex:
        print(f"Could not delete folder {path}")
        print(ex)


def create_directory(path: str):
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


def create_tool_folder(tool_name: str):
    """
    Creates a folder containing all information about the tool, like runtime analysis, plots etc.
    :param tool_name:
    :return:
    """

    path = Path(Runtime_Folders.EVALUATION_DIRECTORY, tool_name)
    try:
        Path(path).mkdir(parents=True, exist_ok=True)

    except OSError as ex:
        print(f"The folder creation for tool {tool_name} failed.")
        print("The tool evaluation will be skipped!")
        # TODO: Add debug mode
        print(ex)
        return None
    else:
        return path


def create_file_folder(tool_path: Path, file_name: str):
    """
    Creates a folder containing all information about the evaluated file, like runtime analysis, plots etc.
    :param file_name:
    :param tool_path:
    :return:
    """

    path = Path(tool_path, file_name)
    try:
        Path(path).mkdir(parents=True, exist_ok=True)

    except OSError as ex:
        print(f"The folder creation for file {file_name} failed.")
        print("The file evaluation will be skipped!")
        # TODO: Add debug mode
        print(ex)
        return None
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
