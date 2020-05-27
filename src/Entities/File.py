from pathlib import Path
import pandas as pd
from Services.FileSystem import Folder_Management, File_Management
from RuntimeContants import Runtime_Folders
import os
from Services.Configuration.Config import Config
from Services.Processing import PreProcessing


class File:
    def __init__(self, path: str, tool_folder: Path):
        self.path = path
        self.name = os.path.splitext(path)[0]
        # The folder where all reports and plots are getting stored
        self.folder = Folder_Management.create_file_folder(tool_folder, self.name)

        if self.folder is not None:
            self.verified = True
        else:
            self.verified = False

        # Stop the init method because the folder is not available and therefore no data can be stored
        if not self.verified:
            return

        # Load data set depending on memory saving mode
        if not Config.MEMORY_SAVING_MODE:
            self.raw_df = File_Management.read_file(self.path)
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

        self.predicted_values = pd.DataFrame(columns=['y', 'y_hat'])

        self.runtime_evaluation_percentage = pd.DataFrame(
            columns=['0', '10', '20', '30', '40', '50', '60', '70', '80',
                     '90', '91', '92', '93', '94', '95', '96', '97', '98',
                     '99', 'Rows', 'Features'])

        self.memory_evaluation_percentage = pd.DataFrame(
            columns=['0', '10', '20', '30', '40', '50', '60', '70', '80',
                     '90', '91', '92', '93', '94', '95', '96', '97', '98',
                     '99', 'Rows', 'Features'])

    def get_column_count(self):
        pass

    def get_row_count(self):
        pass

    def get_feature_count(self):
        pass

    def read_raw_data(self):
        try:
            self.raw_df = pd.read_csv(self.path)
        except OSError as ex:
            print(ex)
            Folder_Management.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)

# EVALUATED_FILE_ROW_COUNT = 0
# EVALUATED_FILE_COLUMN_COUNT = 0
# EVALUATED_FILE_FEATURE_COUNT = 0
# EVALUATED_FILE_RAW_DATA_SET = pd.DataFrame()
# EVALUATED_FILE_PREPROCESSED_DATA_SET = pd.DataFrame()
# EVALUATED_FILE_NO_USEFUL_INFORMATION = False

# General information about a data set with removed rows
# EVALUATED_FILE_REMOVED_ROWS_RUNTIME_INFORMATION = pd.DataFrame(
#   columns=['0', '10', '20', '30', '40', '50', '60', '70', '80',
#           '90', '91', '92', '93', '94', '95', '96', '97', '98',
#          '99', 'Rows', 'Features'])
#
# EVALUATED_FILE_REMOVED_ROWS_MEMORY_INFORMATION = pd.DataFrame(
#   columns=['0', '10', '20', '30', '40', '50', '60', '70', '80',
#             '90', '91', '92', '93', '94', '95', '96', '97', '98',
#            '99', 'Rows', 'Features'])
#
# Predictions vs Real
# EVALUATED_FILE_PREDICTED_VALUES = pd.DataFrame(columns=['y', 'y_hat'])

# General information about runtime predictions using a non modified data set
# EVALUATED_FILE_RUNTIME_INFORMATION = pd.DataFrame(
#   columns=['File Name', 'Train Score', 'Test Score', 'Potential Over Fitting', 'Initial Row Count',
#           'Initial Feature Count', 'Processed Row Count', 'Processed Feature Count'])

# General information about memory predictions using a non modified data set
# EVALUATED_FILE_MEMORY_INFORMATION = pd.DataFrame(
#   columns=['File Name', 'Train Score', 'Test Score', 'Potential Over Fitting', 'Initial Row Count',
#           'Initial Feature Count', 'Processed Row Count', 'Processed Feature Count'])
