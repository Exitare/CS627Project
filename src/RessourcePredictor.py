import argparse
import signal
import sys
from Services import NumpyHelper, Config
from Services.Plotting import Plotting_Full_DS, Plotting_Data_Removal
from Services.Predictions import Single_Predictions, Data_Removal
from Services.File import General_File_Service, Data_Removal
from RuntimeContants import Runtime_Datasets, Runtime_Folders, Runtime_File_Data
from Services import ArgumentParser
import numpy as np


def process_data_sets():
    """
    Processes all data sets, read from the given folder
    :return:
    """

    if len(Runtime_Datasets.RAW_FILE_PATHS) == 0:
        print("No files found to evaluate. Stopping")
        General_File_Service.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
        sys.exit()

    for filename in Runtime_Datasets.RAW_FILE_PATHS:
        try:
            print(f"Evaluating {filename}")
            # Generate tool folder
            General_File_Service.read_file(filename)
            General_File_Service.create_tool_folder(filename)

            # Set important runtime file values
            Runtime_File_Data.EVALUATED_FILE_NAME = filename
            # Reduce len of columns by one, because y value is included
            Runtime_File_Data.EVALUATED_FILE_COLUMN_COUNT = len(
                Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET.columns) - 1
            Runtime_File_Data.EVALUATED_FILE_ROW_COUNT = len(Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET.index)

            # Test for infinite values
            for column in Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET:
                if Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET[column].any() > np.iinfo('i').max:
                    continue

            # Working on full data set
            Single_Predictions.compare_real_to_predicted_data(Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET)

            # Remove data by percentage
            if Runtime_Datasets.COMMAND_LINE_ARGS.remove:
                print("Removing data by percentage")
                evaluate_data_set_by_removing_data(filename, Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET)

        except Exception as ex:
            print("error occurred in process_data_sets()")
            print(ex)
            General_File_Service.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
            sys.exit()


def generate_csv_file():
    """
    Writes all specified data sets
    :return:
    """
    General_File_Service.create_csv_file(Runtime_Datasets.OVER_UNDER_FITTING, Runtime_Folders.CURRENT_WORKING_DIRECTORY,
                                         "Over_Under_Fitting")




def signal_handler(sig, frame):
    """
    Handles a signal. Like pressing crtl +c
    :param sig:
    :param frame:
    :return:
    """
    print('Shutting down gracefully!')
    print("Deleting working directory")
    General_File_Service.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
    print("Done")
    print("Bye")
    sys.exit(0)


def evaluate_data_set_by_removing_data(filename: str, df):
    try:
        if 'runtime' in df.columns:
            print("Predicting runtime...")
            scores = Data_Removal.predict(df, 'runtime')
            input()
            Plotting_Data_Removal.tool_evaluation(scores, "runtime")
            General_File_Service.create_csv_file(scores, Runtime_Datasets.CURRENT_EVALUATED_TOOL_DIRECTORY, "runtime")

            mean_over_file = NumpyHelper.get_mean_per_column_per_df(scores)
            var_over_file = NumpyHelper.get_var_per_column_per_df(scores)

            NumpyHelper.replace_column_with_array(Runtime_Datasets.RUNTIME_MEAN_REPORT, file_index, mean_over_file)
            NumpyHelper.replace_column_with_array(Runtime_Datasets.RUNTIME_VAR_REPORT, file_index, var_over_file)

        if 'memory.max_usage_in_bytes' in df.columns:
            print("Predicting memory...")
            scores = Data_Removal.predict(df, 'memory.max_usage_in_bytes')
            input()
            Plotting_Data_Removal.tool_evaluation(scores, "memory")
            General_File_Service.create_csv_file(scores, Runtime_Datasets.CURRENT_EVALUATED_TOOL_DIRECTORY, "memory")

            mean_over_file = NumpyHelper.get_mean_per_column_per_df(scores)
            var_over_file = NumpyHelper.get_var_per_column_per_df(scores)

            NumpyHelper.replace_column_with_array(Runtime_Datasets.MEMORY_MEAN_REPORT, file_index, mean_over_file)
            NumpyHelper.replace_column_with_array(Runtime_Datasets.MEMORY_VAR_REPORT, file_index, var_over_file)

    except BaseException as ex:
        print(ex)
        General_File_Service.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
        sys.exit()


signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    ArgumentParser.handle_args()
    Config.read_conf()
    General_File_Service.check_folder_integrity()
    General_File_Service.create_evaluation_folder()
    General_File_Service.get_all_file_paths(Config.Config.DATA_RAW_DIRECTORY)
    process_data_sets()
    generate_csv_file()
    #plot_data_sets()
    print("Done")
    exit(0)
