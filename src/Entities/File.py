from pathlib import Path
import pandas as pd
from Services.FileSystem import Folder_Management, File_Management
from RuntimeContants import Runtime_Folders
import os
from Services.Configuration.Config import Config
from Services.Processing import PreProcessing
from time import sleep
import numpy as np
import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


class File:
    def __init__(self, path: str, tool_folder: Path):
        self.path = path
        self.name = os.path.splitext(path)[0]
        self.verified = True

        # Load data set depending on memory saving mode
        if not Config.MEMORY_SAVING_MODE:
            self.raw_df = File_Management.read_file(self.path)
            if self.raw_df is None:
                self.verified = False
                return
        else:
            self.raw_df = pd.DataFrame()

        # Pre process the raw data set
        self.preprocessed_df = PreProcessing.pre_process_data_set(self.raw_df)

        # Setup required data sets
        self.runtime_evaluation = pd.DataFrame(
            columns=['File Name', 'Train Score', 'Test Score', 'Potential Over Fitting', 'Initial Row Count',
                     'Initial Feature Count', 'Processed Row Count', 'Processed Feature Count'])
        self.memory_evaluation = pd.DataFrame(
            columns=['File Name', 'Train Score', 'Test Score', 'Potential Over Fitting', 'Initial Row Count',
                     'Initial Feature Count', 'Processed Row Count', 'Processed Feature Count'])

        self.predicted_runtime_values = pd.DataFrame(columns=['y', 'y_hat'])
        self.predicted_memory_values = pd.DataFrame(columns=['y', 'y_hat'])

        self.runtime_evaluation_percentage = pd.DataFrame(
            columns=['0', '10', '20', '30', '40', '50', '60', '70', '80',
                     '90', '91', '92', '93', '94', '95', '96', '97', '98',
                     '99', 'Rows', 'Features'])

        self.memory_evaluation_percentage = pd.DataFrame(
            columns=['0', '10', '20', '30', '40', '50', '60', '70', '80',
                     '90', '91', '92', '93', '94', '95', '96', '97', '98',
                     '99', 'Rows', 'Features'])

        self.verify_file()

        # Return, because the file is not eligible to be evaluated.
        if not self.verified:
            return

        # The folder where all reports and plots are getting stored, only created if file is valid
        self.folder = Folder_Management.create_file_folder(tool_folder, self.name)

        if self.folder is not None:
            self.verified = True
        else:
            self.verified = False

    def load_data(self):
        """
        Loads the data set. Only used if memory saving mode is active
        :return:
        """
        self.raw_df = File_Management.read_file(self.path)
        self.preprocessed_df = PreProcessing.pre_process_data_set(self.raw_df)

    def get_raw_df_statistics(self):
        """
        Returns column, row and feature count of the raw data set
        :return:
        """
        columns: int = len(self.raw_df.columns)
        rows: int = len(self.raw_df.index)
        features: int = columns - 1
        return columns, rows, features

    def get_pre_processed_df_statistics(self):
        """
        Returns column, row and feature count of the raw data set
        :return:
        """
        columns: int = len(self.preprocessed_df.columns)
        rows: int = len(self.preprocessed_df.index)
        features: int = columns - 1
        return columns, rows, features

    def read_raw_data(self):
        """
        Read the file into memory
        :return:
        """
        try:
            self.raw_df = pd.read_csv(self.path)
        except OSError as ex:
            logging.error(ex)
            self.raw_df = None

    def verify_file(self):
        """
        Check if the file passes all requirements to be able to be evaluated
        :return:
        """
        columns, rows, features = self.get_raw_df_statistics()
        if rows < Config.MINIMUM_ROW_COUNT:
            logging.warning(f"{self.name} has not sufficient rows ({rows}).")
            logging.warning("The file will not be evaluated.")
            sleep(1)
            self.verified = False

        if columns < Config.MINIMUM_COLUMN_COUNT:
            logging.warning(f"{self.name} has not sufficient columns ({columns}).")
            logging.warning("The file will not be evaluated.")
            sleep(1)
            self.verified = False

        # check for infinity values
        for column in self.preprocessed_df:
            if self.preprocessed_df[column].any() > np.iinfo('i').max:
                logging.warning(f"Detected infinity values in preprocessed data set!")
                logging.warning(f"File will not be evaluated.")
                self.verified = False

    def predict_runtime(self):
        """
        Predicts the runtime for a complete data set.
        :return:
        """
        df = self.preprocessed_df.copy()

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
            {'File Name': self.name, "Test Score": test_score,
             "Train Score": train_score, "Potential Over Fitting": over_fitting,
             "Initial Row Count": len(self.raw_df.index),
             "Initial Feature Count": len(self.raw_df.columns) - 1, "Processed Row Count": len(X),
             "Processed Feature Count": X.shape[1]}, ignore_index=True)

        self.predicted_runtime_values = pd.concat([pd.Series(y_test).reset_index()['runtime'], pd.Series(y_test_hat)],
                                                  axis=1)

    def predict_memory(self):
        """
        Predicts the memory usage for a complete data set.
        :return:
        """
        df = self.preprocessed_df.copy()

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
            {'File Name': self.name, "Test Score": test_score,
             "Train Score": train_score, "Potential Over Fitting": over_fitting,
             "Initial Row Count": len(self.raw_df.index),
             "Initial Feature Count": len(self.raw_df.columns) - 1, "Processed Row Count": len(X),
             "Processed Feature Count": X.shape[1]}, ignore_index=True)

        self.predicted_memory_values = pd.concat(
            [pd.Series(y_test).reset_index()['memory.max_usage_in_bytes'], pd.Series(y_test_hat)],
            axis=1)

    def free_memory(self):
        """
        Release not required memory for memory saving mode.
        :return:
        """
        if not Config.MEMORY_SAVING_MODE:
            return

        self.raw_df = None
        self.preprocessed_df = None
