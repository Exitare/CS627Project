import pandas as pd
from pathlib import Path
from Entities.File import File
from RuntimeContants import Runtime_Folders
from Services.FileSystem import Folder_Management
from Services.Configuration.Config import Config


class Tool:
    def __init__(self, name: str):
        self.name = name
        self.combined_data_set = pd.DataFrame()
        # Timestamped folder for this specific run of the application.
        self.evaluation_dir = Runtime_Folders.EVALUATION_DIRECTORY
        # the tool folder
        self.folder = Folder_Management.create_tool_folder(self.name)
        self.all_files = []
        self.excluded_files = []
        self.verified_files = []

        # if all checks out, the tool will be flag as verified
        # Tools flagged as not verified will not be evaluated
        if self.folder is not None:
            self.verified = True
        else:
            self.verified = False

    def __eq__(self, other):
        """
        Checks if another tool entity is equal to this one
        :param other:
        :return:
        """
        return self.name == other.name

    def add_file(self, file_path: str):
        """
        Adds a new file entity to the tool
        :param file_path:
        :return:
        """
        file: File = File(file_path, self.folder)
        self.all_files.append(file)

    def verify(self):
        """
        Checks if the tool contains actual files
        :return:
        """
        if len(self.all_files) == 0:
            Folder_Management.remove_folder(self.folder)
            self.verified = False
            print(f"Tool {self.name} does not contain any files. The tool will not evaluated.")

        self.verified_files = [file for file in self.all_files if file.verified]
        self.excluded_files = [file for file in self.all_files if not file.verified]

        if Config.DEBUG_MODE:
            print(
                f"Tool contains {len(self.excluded_files)} exlcuded files and {len(self.verified_files)} verified files.")

        if len(self.verified_files) == 0:
            Folder_Management.remove_folder(self.folder)
            self.verified = False
            print(f"Tool {self.name} does not contain at least one file that is verified. The tool will not evaluated.")

    def evaluate_files(self):
        for file in self.files:

            if Config.VERBOSE:
                print(f"Evaluating {file_path}")
