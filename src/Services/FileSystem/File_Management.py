from pathlib import Path
import ntpath
import os
import sys
from RuntimeContants import Runtime_Folders, Runtime_Datasets, Runtime_File_Data
from Services.Configuration.Config import Config
from Services.FileSystem import Folder_Management
import pandas as pd
from collections import defaultdict
from Entities import File
from Entities.Tool import Tool
from Entities.File import File


def load_tools():
    """
    Gathers all data, which is required. Takes command line args into account.
    :return:
    """

    for file in os.listdir(Config.DATA_RAW_DIRECTORY):
        file_path = os.fsdecode(file)
        try:
            if file_path.endswith(".csv") or file_path.endswith(".tsv"):
                file_name: str = get_file_name(file_path)
                tool_name: str = str(file_name.rsplit('_', 1)[0])
                tool = Tool(tool_name)

                tool_found: bool = False
                for stored_tool in Runtime_Datasets.DETECTED_TOOLS:
                    if not tool_found:
                        if stored_tool.__eq__(tool):
                            stored_tool.add_file(file_path)
                            tool_found = True
                            break

                if tool_found:
                    continue
                else:
                    if Config.VERBOSE:
                        print(f"Added tool {tool.name}")
                    tool.add_file(file_path)
                    Runtime_Datasets.DETECTED_TOOLS.append(tool)
        except OSError as ex:
            print(ex)
        except BaseException as ex:
            print(ex)

    for tool in Runtime_Datasets.DETECTED_TOOLS:
        for file in tool.files:
            print(f"Path {file.path}")
            print(f"Name {file.name}")


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
        if Config.VERBOSE:
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
