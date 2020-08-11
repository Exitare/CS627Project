import pandas as pd
from Entities.File import File
from RuntimeContants import Runtime_Folders
from Services.FileSystem import Folder_Management
from Services.Configuration.Config import Config
from pathlib import Path
import logging
from time import sleep
import os
import seaborn as sns
import shutil


class Tool:
    def __init__(self, name: str):
        self.name = name
        self.combined_data_set = pd.DataFrame()
        # Timestamped folder for this specific run of the application.
        self.evaluation_dir = Runtime_Folders.EVALUATION_DIRECTORY
        # the tool folder
        self.folder = Folder_Management.create_tool_folder(self.name)
        self.all_files = []
        # All files not eligible to be checked
        self.excluded_files = []
        # All files eligible to be evaluated
        self.verified_files = []

        self.files_runtime_overview = pd.DataFrame()
        self.files_memory_overview = pd.DataFrame()

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
        Checks if the tool contains actual files and is therefore a valid /verified tool
        If no file is associated to the tool, the tool folder will be deleted.
        If more than one file is associated to the tool a merged file will be added
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

        # Add a merged file to the tool.
        if len(self.verified_files) > 1:
            self.__add_merged_file()

    def free_memory(self):
        """
        Frees memory if the memory saving mode is active
        :return:
        """

        if not Config.MEMORY_SAVING_MODE:
            return

        logging.info("Cleaning memory...")
        for file in self.verified_files:
            file.free_memory()

        sleep(1)

    def evaluate(self):
        """
        Handles the evaluation of a tool
        :return:
        """
        print()
        logging.info(f"Evaluating tool {self.name}...")
        # Load data for each file of the tool because it was not loaded at the start
        if Config.MEMORY_SAVING_MODE:
            for file in self.verified_files:
                file.load_raw_data()
                file.verify()

        # Evaluate the files
        self.__evaluate_verified_files()

    def evaluate_additional_files(self):
        """
        Evaluate all data sets and files which are create after the first evaluation
        """
        for file in self.verified_files:
            if file.evaluated:
                continue

            logging.info(f"Evaluating file {file.name}...")

            # Predict values for single files
            file.predict(Config.RUNTIME_LABEL)
            file.predict(Config.MEMORY_LABEL)
            if Config.PERCENTAGE_REMOVAL:
                file.predict_row_removal(Config.RUNTIME_LABEL)
                file.predict_row_removal(Config.MEMORY_LABEL)

            file.evaluated = True

    def prepare_additional_files(self):
        """
        Prepare additional files after the first evaluation. E.g. merge only best performing versions instead of all.
        """
        if len(self.verified_files) <= 1:
            return

        print()
        logging.info("Preparing additional files...")
        self.__prepare_best_performing_version_merged_file()

    def __prepare_best_performing_version_merged_file(self):
        """
        Prepares a merged data set which contains only data from version with a test score > 0.6
        """
        # Create merged files containing only data sets of version with a Test Score greated than 0.6
        best_performing = self.files_runtime_overview[self.files_runtime_overview['Test Score'] > 0.6][
            'File Name'].tolist()

        # Create a merged data set containing only versions which are performing good
        best_versions_df = []
        for file in self.verified_files:
            if file.name in best_performing and not file.merged_file:
                best_versions_df.append(file.raw_df)

        if len(best_versions_df) <= 1:
            return

        best_version_files_raw_df = pd.concat(best_versions_df, join='inner')
        best_version_merged_file = File("best_version_merged_file", self.folder, best_version_files_raw_df)
        self.verified_files.append(best_version_merged_file)

    def __prepare_most_important_feature_data_set(self):
        """
        Generates a data set for each version which contains only the most important features for each version
        """

        for file in self.verified_files:
            if file.merged_file:
                continue

    def generate_reports(self):
        """
        Generate csv and tsv files for the tool and all files associated to the tool
        :return:
        """
        logging.info("Generating report files...")

        # Generate file specific reports
        for file in self.verified_files:
            file.generate_reports()

        if not self.files_runtime_overview.empty:
            self.files_runtime_overview.sort_values(by='Test Score', ascending=False, inplace=True)
            self.files_runtime_overview.to_csv(os.path.join(self.folder, "overview_files_runtime_report.csv"),
                                               index=False)

        if not self.files_memory_overview.empty:
            self.files_memory_overview.sort_values(by='Test Score', ascending=False, inplace=True)
            self.files_memory_overview.to_csv(os.path.join(self.folder, "overview_files_memory_report.csv"),
                                              index=False)

        logging.info("All reports generated.")
        sleep(1)

    def generate_plots(self):
        """
        Generates all plots
        :return:
        """

        # Generate plots for each file associated to the tool
        for file in self.verified_files:
            file.generate_plots()

    def generate_overview_data_sets(self):
        """
        Creates the overview data sets
        """
        self.files_runtime_overview = pd.DataFrame()
        self.files_memory_overview = pd.DataFrame()

        for file in self.verified_files:
            self.files_runtime_overview = self.files_runtime_overview.append(file.runtime_evaluation)
            self.files_memory_overview = self.files_memory_overview.append(file.memory_evaluation)

    def __evaluate_verified_files(self):
        """
        Evaluates all files associated to a tool.
        Runtime and memory is evaluated
        :return:
        """

        for file in self.verified_files:
            logging.info(f"Evaluating file {file.name}...")

            # Predict values for single files
            file.predict(Config.RUNTIME_LABEL)
            file.predict(Config.MEMORY_LABEL)
            if Config.PERCENTAGE_REMOVAL:
                file.predict_row_removal(Config.RUNTIME_LABEL)
                file.predict_row_removal(Config.MEMORY_LABEL)

            # Copy the source file to the results folder
            # If its a merged file use the virtual one.
            if not file.merged_file:
                shutil.copy(file.path, file.folder)
            else:
                file.raw_df.to_csv(os.path.join(file.folder, "raw_df.csv"), index=False)

            file.evaluated = True

    def __add_merged_file(self):
        """
        Merges the raw data sets of all verified versions into a big one.
        Assuming that all single files are valid this merged on should be valid too.
        :return:
        """
        raw_df = []
        for file in self.verified_files:
            raw_df.append(file.raw_df)

        merged_files_raw_df = pd.concat(raw_df, join='inner')
        merged_file = File("merged_tool", self.folder, merged_files_raw_df)
        self.verified_files.append(merged_file)

    def get_best_performing_version(self, runtime: bool):
        """
        Returns the best performing version of the tool
        """
        if runtime:
            if self.files_runtime_overview.empty:
                return None

            self.files_runtime_overview = self.files_runtime_overview.reset_index()
            row_id = self.files_runtime_overview['Test Score'].argmax()
            return self.files_runtime_overview.loc[row_id]
        else:
            if self.files_memory_overview.empty:
                return None

            self.files_memory_overview = self.files_memory_overview.reset_index()
            row_id = self.files_memory_overview['Test Score'].argmax()
            return self.files_memory_overview.loc[row_id]

    def get_worst_performing_version(self, runtime: bool):
        """
        Returns the worst performing version of the tool
        """
        if runtime:
            if self.files_runtime_overview.empty:
                return None

            self.files_runtime_overview = self.files_runtime_overview.reset_index()
            row_id = self.files_runtime_overview['Test Score'].argmin()
            return self.files_runtime_overview.loc[row_id]
        else:
            if self.files_memory_overview.empty:
                return None

            self.files_memory_overview = self.files_memory_overview.reset_index()
            row_id = self.files_memory_overview['Test Score'].argmin()
            return self.files_memory_overview.loc[row_id]
