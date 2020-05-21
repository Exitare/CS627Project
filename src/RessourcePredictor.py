import argparse
import signal
import sys
from Services import Config, PreProcessing
from Services.Predictions import Single_Predictions, Predict_Data_Removal
from Services.File import General_File_Service
from RuntimeContants import Runtime_Datasets, Runtime_Folders, Runtime_File_Data
from Services.Plotting import Plotting_Data_Removal
from Services import ArgumentParser
import numpy as np
import pandas as pd
from Services import Misc


def process_data_sets():
    """
    Processes all data sets, read from the given folder
    :return:
    """

    if len(Runtime_Datasets.RAW_FILE_PATHS) == 0:
        print("No files found to evaluate. Stopping")
        General_File_Service.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
        sys.exit()

    # Get similar files in directory
    General_File_Service.get_similar_files()

    # merged file prediction
    for file, path in Runtime_Datasets.RAW_SIMILAR_FILES.items():
        data_frames = []

        for filename in path:
            General_File_Service.read_file(filename)
            data_frames.append(Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET)

        merged_df = pd.concat(data_frames)
        print(f"Evaluating {file}")
        Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET = merged_df
        General_File_Service.create_tool_folder(file)
        PreProcessing.pre_process_data_set(Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET)
        Misc.set_general_file_data(file)

        if Runtime_File_Data.EVALUATED_FILE_ROW_COUNT < Config.Config.MINIMUM_ROW_COUNT:
            if Runtime_Datasets.COMMAND_LINE_ARGS.verbose:
                print(f"Data set {file} has insufficient row count and will be ignored. ")

            Runtime_Datasets.EXCLUDED_FILES = Runtime_Datasets.EXCLUDED_FILES.append(
                {'File': file, 'Rows': Runtime_File_Data.EVALUATED_FILE_ROW_COUNT}, ignore_index=True)
            General_File_Service.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
            continue

        Single_Predictions.compare_real_to_predicted_data()

        if Runtime_Datasets.COMMAND_LINE_ARGS.remove:
            if Runtime_File_Data.EVALUATED_FILE_NO_USEFUL_INFORMATION:
                print(
                    f"File {Runtime_File_Data.EVALUATED_FILE_NAME} does not contain useful information. Skipping...")
                continue
            print("Removing data by percentage")
            Predict_Data_Removal.removal_helper()

        generate_file_report_files()

    # Single file prediction
    for filename in Runtime_Datasets.RAW_FILE_PATHS:
        try:
            print(f"Evaluating {filename}")
            # Generate tool folder
            General_File_Service.create_tool_folder(filename)
            # Load the data set
            General_File_Service.read_file(filename)
            # Pre process the df
            PreProcessing.pre_process_data_set(Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET)
            Misc.set_general_file_data(filename)

            # Test for infinite values
            for column in Runtime_File_Data.EVALUATED_FILE_PREPROCESSED_DATA_SET:
                if Runtime_File_Data.EVALUATED_FILE_PREPROCESSED_DATA_SET[column].any() > np.iinfo('i').max:
                    continue

            # Check if dataset row count is equal or greater compared to the treshold set in the config
            if Runtime_File_Data.EVALUATED_FILE_ROW_COUNT < Config.Config.MINIMUM_ROW_COUNT:
                if Runtime_Datasets.COMMAND_LINE_ARGS.verbose:
                    print(f"Data set {filename} has insufficient row count and will be ignored. ")

                General_File_Service.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
                Runtime_Datasets.EXCLUDED_FILES = Runtime_Datasets.EXCLUDED_FILES.append(
                    {'File': filename, 'Rows': Runtime_File_Data.EVALUATED_FILE_ROW_COUNT}, ignore_index=True)
                continue

            # Working on full data set
            print("Evaluation full data set")
            Single_Predictions.compare_real_to_predicted_data()

            # Remove data by percentage
            if Runtime_Datasets.COMMAND_LINE_ARGS.remove:
                if Runtime_File_Data.EVALUATED_FILE_NO_USEFUL_INFORMATION:
                    if Runtime_Datasets.COMMAND_LINE_ARGS.verbose:
                        print(f"File {Runtime_File_Data.EVALUATED_FILE_NAME}"
                              f"does not contain useful information. Skipping...")
                    continue

                print("Removing data by percentage")
                Predict_Data_Removal.removal_helper()

            generate_file_report_files()

        except Exception as ex:
            print("error occurred in process_data_sets()")
            print(ex)
            General_File_Service.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
            sys.exit()


def generate_file_report_files():
    # Write general information about the data set
    General_File_Service.create_csv_file(Runtime_File_Data.EVALUATED_FILE_RUNTIME_INFORMATION,
                                         Runtime_Folders.CURRENT_WORKING_DIRECTORY,
                                         "General_Information_Runtime")
    General_File_Service.create_csv_file(Runtime_File_Data.EVALUATED_FILE_MEMORY_INFORMATION,
                                         Runtime_Folders.CURRENT_WORKING_DIRECTORY,
                                         "General_Information_Memory")
    # Write general information for a specific tool , non modified
    General_File_Service.create_csv_file(Runtime_File_Data.EVALUATED_FILE_REMOVED_ROWS_RUNTIME_INFORMATION,
                                         Runtime_Folders.CURRENT_EVALUATED_TOOL_DIRECTORY,
                                         "data_removal_runtime_evaluation")
    General_File_Service.create_csv_file(Runtime_File_Data.EVALUATED_FILE_REMOVED_ROWS_MEMORY_INFORMATION,
                                         Runtime_Folders.CURRENT_EVALUATED_TOOL_DIRECTORY,
                                         "data_removal_memory_evaluation")


def generate_generate_report_files():
    """
    Writes all specified data sets
    :return:
    """

    # Write the mean and var reports for all files
    General_File_Service.create_csv_file(Runtime_Datasets.RUNTIME_VAR_REPORT,
                                         Runtime_Folders.CURRENT_WORKING_DIRECTORY,
                                         Config.Config.FILE_RUNTIME_VAR_SUMMARY)

    General_File_Service.create_csv_file(Runtime_Datasets.RUNTIME_MEAN_REPORT,
                                         Runtime_Folders.CURRENT_WORKING_DIRECTORY,
                                         Config.Config.FILE_RUNTIME_MEAN_SUMMARY)

    General_File_Service.create_csv_file(Runtime_Datasets.MEMORY_MEAN_REPORT,
                                         Runtime_Folders.CURRENT_WORKING_DIRECTORY,
                                         Config.Config.FILE_MEMORY_MEAN_SUMMARY)

    General_File_Service.create_csv_file(Runtime_Datasets.MEMORY_VAR_REPORT,
                                         Runtime_Folders.CURRENT_WORKING_DIRECTORY,
                                         Config.Config.FILE_MEMORY_VAR_SUMMARY)

    General_File_Service.create_csv_file(Runtime_Datasets.EXCLUDED_FILES, Runtime_Folders.CURRENT_WORKING_DIRECTORY,
                                         Config.Config.FILE_EXCLUDED_FILES)


def plot_data_sets():
    Plotting_Data_Removal.tool_evaluation(Runtime_File_Data.EVALUATED_FILE_REMOVED_ROWS_RUNTIME_INFORMATION, "runtime")
    Plotting_Data_Removal.tool_evaluation(Runtime_File_Data.EVALUATED_FILE_REMOVED_ROWS_MEMORY_INFORMATION, "memory")


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


signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    ArgumentParser.handle_args()
    Config.read_conf()
    General_File_Service.check_folder_integrity()
    General_File_Service.create_evaluation_folder()
    General_File_Service.get_all_file_paths(Config.Config.DATA_RAW_DIRECTORY)
    process_data_sets()
    generate_generate_report_files()
    plot_data_sets()
    print("Done")
    exit(0)
