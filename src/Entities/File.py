from pathlib import Path
import pandas as pd
from Services.FileSystem import Folder_Management, File_Management
from RuntimeContants import Runtime_Folders
import os


class File:
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.splitext(path)[0]
        self.result_directory = Folder_Management.create_tool_folder(self.name)
        self.raw_df = pd.DataFrame()
        self.preprocessed_df = pd.DataFrame()
        self.runtime_evaluation = pd.DataFrame(
            columns=['File Name', 'Train Score', 'Test Score', 'Potential Over Fitting', 'Initial Row Count',
                     'Initial Feature Count', 'Processed Row Count', 'Processed Feature Count'])
        self.memory_evaluation = pd.DataFrame(
            columns=['File Name', 'Train Score', 'Test Score', 'Potential Over Fitting', 'Initial Row Count',
                     'Initial Feature Count', 'Processed Row Count', 'Processed Feature Count'])

        self.predicted_values = pd.DataFrame(columns=['y', 'y_hat'])

        self.removed_rows_runtime_evaluation = pd.DataFrame(
            columns=['0', '10', '20', '30', '40', '50', '60', '70', '80',
                     '90', '91', '92', '93', '94', '95', '96', '97', '98',
                     '99', 'Rows', 'Features'])

        self.removed_rows_memory_evaluation = pd.DataFrame(
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
