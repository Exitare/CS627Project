import datetime
from pathlib import Path
from Services.Configuration.Config import Config
from RuntimeContants import Runtime_Folders
import sys
import shutil
import logging

folder_management = logging.getLogger()
folder_management.setLevel(logging.DEBUG)


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
        folder_management.warning(f"Could not create evaluation directory {path}")
        folder_management.warning("Stopping application")
        if Config.DEBUG_MODE:
            folder_management.warning(ex)
        sys.exit()
    else:
        Runtime_Folders.EVALUATION_DIRECTORY = path


def remove_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as ex:
        folder_management.warning(f"Could not delete folder {path}")
        if Config.DEBUG_MODE:
            folder_management.warning(ex)


def create_directory(path: str):
    """
    Creates a folder with the given path
    :param path:
    :return:
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)

    except OSError as ex:
        folder_management.critical(ex)
        folder_management.critical(f"Creation of the directory {path} failed.")
        folder_management.critical("Stopping application")
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
        folder_management.warning(f"The folder creation for tool {tool_name} failed.")
        folder_management.warning("The tool evaluation will be skipped!")
        if Config.DEBUG_MODE:
            folder_management.warning(ex)
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
        return path
    except BaseException as ex:
        logging.warning(f"The folder creation for file {file_name} failed.")
        logging.warning("The file evaluation will be skipped!")
        if Config.DEBUG_MODE:
            logging.warning(ex)
        return None


def create_required_folders():
    """
    Checks if the Data folder structure is up to date given the options from the config
    :return:
    """
    created: bool = False

    folder_management.info("Checking data folder integrity...")
    if not Config.DATA_ROOT_DIRECTORY.is_dir():
        created = True
        folder_management.info("Root directory not found. Creating...")
        create_directory(Config.DATA_ROOT_DIRECTORY)

    if not Config.DATA_RAW_DIRECTORY.is_dir():
        created = True
        folder_management.info("Raw data directory not found. Creating...")
        create_directory(Config.DATA_RAW_DIRECTORY)

    if not Config.DATA_RESULTS_DIRECTORY.is_dir():
        created = True
        folder_management.info("Results data directory not found. Creating...")
        create_directory(Config.DATA_RESULTS_DIRECTORY)

    return created
