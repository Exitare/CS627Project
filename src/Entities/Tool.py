import pandas as pd
from pathlib import Path
from Entities.File import File, PredictiveColumn
from RuntimeContants import Runtime_Folders
from Services.FileSystem import Folder_Management, File_Management
from Services.Configuration.Config import Config
from Services.Processing import PreProcessing
import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from time import sleep
import os
import seaborn as sns


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

        self.merged_files_raw_df = pd.DataFrame()
        self.merged_files_preprocessed_df = pd.DataFrame()
        # Setup required data sets
        self.runtime_evaluation = pd.DataFrame(
            columns=['File Name', 'Train Score', 'Test Score', 'Potential Over Fitting', 'Initial Row Count',
                     'Initial Feature Count', 'Processed Row Count', 'Processed Feature Count'])
        self.memory_evaluation = pd.DataFrame(
            columns=['File Name', 'Train Score', 'Test Score', 'Potential Over Fitting', 'Initial Row Count',
                     'Initial Feature Count', 'Processed Row Count', 'Processed Feature Count'])

        self.predicted_runtime_values = pd.DataFrame(columns=['y', 'y_hat'])
        self.predicted_memory_values = pd.DataFrame(columns=['y', 'y_hat'])

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

    def free_memory(self):
        """
        Frees memory if the memory saving mode is active
        :return:
        """

        if not Config.MEMORY_SAVING_MODE:
            return

        logging.info("Freeing up memory...")
        for file in self.verified_files:
            file.free_memory()

        self.merged_files_raw_df = None
        self.merged_files_preprocessed_df = None

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
                file.load_data()

        # Evaluate the files
        self.__evaluate_merged_df()
        self.__evaluate_verified_files()

    def generate_reports(self):
        """
        Generate csv and tsv files for the tool and all files associated to the tool
        :return:
        """
        logging.info("Generating report files...")

        # Generate file specific reports
        for file in self.verified_files:
            file.generate_reports()

        # Generate file overview reports
        files_runtime_overview = pd.DataFrame()
        files_memory_overview = pd.DataFrame()
        for file in self.verified_files:
            files_runtime_overview = files_runtime_overview.append(file.runtime_evaluation)
            files_memory_overview = files_memory_overview.append(file.memory_evaluation)

        # Generate tool specific reports
        if not self.runtime_evaluation.empty:
            files_runtime_overview = files_runtime_overview.append(self.runtime_evaluation)

        if not self.memory_evaluation.empty:
            files_memory_overview = files_memory_overview.append(self.memory_evaluation)

        if not files_runtime_overview.empty:
            files_runtime_overview.to_csv(os.path.join(self.folder, "files_runtime_report.csv"), index=False)

        if not files_memory_overview.empty:
            files_memory_overview.to_csv(os.path.join(self.folder, "files_memory_report.csv"), index=False)

        if not self.predicted_memory_values.empty:
            self.predicted_memory_values.to_csv(os.path.join(self.folder, "predicted_memory_report.csv"),
                                                index=False)
            self.predicted_memory_values.to_csv(os.path.join(self.folder, "predicted_memory_report.tsv"),
                                                index=False, sep=",")

        if not self.predicted_runtime_values.empty:
            self.predicted_runtime_values.to_csv(os.path.join(self.folder, "predicted_runtime_report.csv"),
                                                 index=False)
            self.predicted_runtime_values.to_csv(os.path.join(self.folder, "predicted_memory_report.tsv"),
                                                 index=False, sep=",")

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

        ax = None

        if not self.predicted_memory_values.empty:
            ax = sns.scatterplot(x='y', y='y_hat', label="memory", data=self.predicted_memory_values)
            ax.set(xscale="log", yscale="log")

        if not self.predicted_runtime_values.empty:
            ax = sns.scatterplot(x='y', y='y_hat', label="runtime", data=self.predicted_runtime_values)
            ax.set(xscale="log", yscale="log")

        ax.legend()
        fig = ax.get_figure()
        fig.savefig(os.path.join(self.folder, "predicated_values.png"))
        fig.clf()

    def __evaluate_verified_files(self):
        """
        Evaluates all files associated to a tool.
       Runtime and memory is evaluated
       :return:
       """

        for file in self.verified_files:
            if Config.VERBOSE:
                logging.info(f"Evaluating file {file.name}...")

            # Predict values for single files
            file.predict_runtime()
            file.predict_memory()
            if Config.PERCENTAGE_REMOVAL:
                file.predict_row_removal(PredictiveColumn.RUNTIME.value)
                file.predict_row_removal(PredictiveColumn.MEMORY.value)

    def __evaluate_merged_df(self):
        """
        Evaluates the merged data set
        :return:
        """

        # Merge files now, because it was not done at the start because of saving memory
        if Config.MEMORY_SAVING_MODE:
            self.merge_files()

        self.__predict_runtime()
        self.__predict_memory()

    def merge_files(self):
        """
        Merges the raw data sets of all files.
        :return:
        """

        data_frames = []
        for file in self.verified_files:
            data_frames.append(file.raw_df)

        self.merged_files_raw_df = pd.concat(data_frames)
        self.merged_files_preprocessed_df = PreProcessing.pre_process_data_set(self.merged_files_raw_df)

    def __predict_runtime(self):
        """
        Predicts the runtime for a complete data set.
        :return:
        """
        df = self.merged_files_preprocessed_df.copy()

        if 'runtime' not in df:
            return

        model = RandomForestRegressor(n_estimators=Config.FOREST_ESTIMATORS, random_state=1)

        y = df['runtime']
        del df['runtime']
        X = df

        source_row_count = len(X)

        X_indexes = (X != 0).any(axis=1)

        X = X.loc[X_indexes]
        y = y.loc[X_indexes]

        if source_row_count != len(X) and Config.VERBOSE:
            logging.info(f"Removed {source_row_count - len(X)} rows. Source had {source_row_count}.")

        X = PreProcessing.normalize_X(X)
        X = PreProcessing.variance_selection(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)

        model.fit(X_train, y_train)
        y_test_hat = model.predict(X_test)
        y_train_hat = model.predict(X_train)
        test_score = r2_score(y_test, y_test_hat)
        train_score = r2_score(y_train, y_train_hat)

        over_fitting = False
        if train_score > test_score * 2:
            over_fitting = True

        self.runtime_evaluation = self.runtime_evaluation.append(
            {'File Name': f"{self.name}_merged", "Test Score": test_score,
             "Train Score": train_score, "Potential Over Fitting": over_fitting,
             "Initial Row Count": len(self.merged_files_raw_df.index),
             "Initial Feature Count": len(self.merged_files_raw_df.columns) - 1, "Processed Row Count": len(X),
             "Processed Feature Count": X.shape[1]}, ignore_index=True)

        self.predicted_runtime_values = pd.concat([pd.Series(y_test).reset_index()['runtime'], pd.Series(y_test_hat)],
                                                  axis=1)
        self.predicted_runtime_values.rename(columns={"runtime": "y", 0: "y_hat"}, inplace=True)

    def __predict_memory(self):
        """
        Predicts the memory usage for a complete data set.
        :return:
        """
        df = self.merged_files_preprocessed_df.copy()

        if 'memory.max_usage_in_bytes' not in df:
            return

        model = RandomForestRegressor(n_estimators=Config.FOREST_ESTIMATORS, random_state=1)

        y = df['memory.max_usage_in_bytes']
        del df['memory.max_usage_in_bytes']
        X = df

        source_row_count = len(X)

        X_indexes = (X != 0).any(axis=1)

        X = X.loc[X_indexes]
        y = y.loc[X_indexes]

        if source_row_count != len(X) and Config.VERBOSE:
            logging.info(f"Removed {source_row_count - len(X)} rows. Source had {source_row_count}.")

        X = PreProcessing.normalize_X(X)
        X = PreProcessing.variance_selection(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)

        model.fit(X_train, y_train)
        y_test_hat = model.predict(X_test)
        y_train_hat = model.predict(X_train)
        test_score = r2_score(y_test, y_test_hat)
        train_score = r2_score(y_train, y_train_hat)

        over_fitting = False
        if train_score > test_score * 2:
            over_fitting = True

        self.memory_evaluation = self.memory_evaluation.append(
            {'File Name': f"{self.name}_merged", "Test Score": test_score,
             "Train Score": train_score, "Potential Over Fitting": over_fitting,
             "Initial Row Count": len(self.merged_files_raw_df.index),
             "Initial Feature Count": len(self.merged_files_raw_df.columns) - 1, "Processed Row Count": len(X),
             "Processed Feature Count": X.shape[1]}, ignore_index=True)

        self.predicted_memory_values = pd.concat(
            [pd.Series(y_test).reset_index()['memory.max_usage_in_bytes'], pd.Series(y_test_hat)],
            axis=1)

        self.predicted_memory_values.rename(columns={"runtime": "y", 0: "y_hat"}, inplace=True)
