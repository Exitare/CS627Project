import pandas as pd
from pathlib import Path
from Entities.File import File
from RuntimeContants import Runtime_Folders
from Services.FileSystem import Folder_Management, File_Management
from Services.Configuration.Config import Config
from Services.Processing import PreProcessing
import logging


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

        self.merged_files_df = []

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
            logging.info(f"Tool {self.name} does not contain any files. The tool will not evaluated.")

        self.verified_files = [file for file in self.all_files if file.verified]
        self.excluded_files = [file for file in self.all_files if not file.verified]

        if Config.DEBUG_MODE:
            logging.info(
                f"Tool contains {len(self.excluded_files)} excluded files and {len(self.verified_files)} verified files.")

        if len(self.verified_files) == 0:
            Folder_Management.remove_folder(self.folder)
            self.verified = False
            logging.info(f"Tool {self.name} does not contain at least one file that is verified")
            logging.info(f"The tool will not evaluated and the folder will be cleanup up.")

    def evaluate_verified_files(self):
        """
        Evaluates all files associated to a tool.
        Runtime and memory is evaluated
        :return:
        """

        for file in self.verified_files:
            if Config.VERBOSE:
                logging.info(f"Evaluating {file.name}")

            if Config.MEMORY_SAVING_MODE:
                if Config.VERBOSE:
                    print(f"Loading data set because of memory saving mode")
                file.raw_df = File_Management.read_file(file.path)
                file.preprocessed_df = PreProcessing.pre_process_data_set(file.raw_df)

            # Predict values for single files
            file.predict_runtime()
            file.predict_memory()

            # Predict merged data sets
            # TODO: Add function

            # Predict percentage removal
            # TODO: ADD function

            # Frees memory if in memory saving mode
            file.free_memory()

    def evaluate_merged_df(self):
        pass

    def evaluate_verified_files_with_percentage(self):
        pass

    def merge_files(self):
        """

        :return:
        """
        if Config.MEMORY_SAVING_MODE:
            return

        data_frames = []

        for filename in path:
            File_Management.read_file(filename)
            data_frames.append(Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET)

        merged_df = pd.concat(data_frames)

        for file in self.verified_files:
            if Config.MEMORY_SAVING_MODE:
                if Config.VERBOSE:
                    print(f"Loading data set because of memory saving mode")
                file.raw_df = File_Management.read_file(file.path)
                file.preprocessed_df = PreProcessing.pre_process_data_set(file.raw_df)
            pass
