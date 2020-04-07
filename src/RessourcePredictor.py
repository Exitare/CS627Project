import argparse
import signal
import sys
from Services import NumpyHelper, Config, Plotting
from Services.Predictions import Single_Predictions, Data_Removal
from Services.File import General_File_Service, Data_Removal
from RuntimeContants import Runtime_Datasets, Runtime_Folders, Runtime_File_Data
from Services import ArgumentParser


def process_data_sets():
    """
    Processes all data sets, read from the given folder
    :return:
    """

    if len(Runtime_Datasets.RAW_FILE_DATA_SETS) == 0:
        print("No files found to evaluate. Stopping")
        General_File_Service.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
        sys.exit()

    for filename, df in Runtime_Datasets.RAW_FILE_DATA_SETS.items():
        try:
            # Generate tool folder
            General_File_Service.create_tool_folder(filename)

            # Set important runtime file values
            Runtime_File_Data.EVALUATED_FILE_NAME = filename
            Runtime_File_Data.EVALUATED_FILE_COLUMN_COUNT = len(df.columns)
            Runtime_File_Data.EVALUATED_FILE_ROW_COUNT = len(df.index)

            # Working on full data set
            Single_Predictions.compare_real_to_predicted_data(df)

            if Runtime_Datasets.COMMAND_LINE_ARGS.remove:
                remove_data(filename, df)

        except Exception as ex:
            print("error occurred in start()")
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

    if Runtime_Datasets.COMMAND_LINE_ARGS.remove:
        Plotting.plot_summary()
        Data_Removal.write_summary()
        Plotting.plot_group_by_parameter_count()


def plot_data_sets():
    """
    Plots all specified data sets
    :return:
    """
    if Runtime_Datasets.COMMAND_LINE_ARGS.remove:
        Plotting.plot_summary()
        Plotting.plot_group_by_parameter_count()


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


def remove_data(filename: str, df):
    try:
        print(f"Evaluating {filename}")
        if 'runtime' in df.columns:
            print("Predicting runtime...")
            scores = Data_Removal.predict(df, 'runtime')
            Plotting.tool_evaluation(scores, "runtime")
            General_File_Service.create_csv_file(scores, Runtime_Datasets.CURRENT_EVALUATED_TOOL_DIRECTORY, "runtime")

            mean_over_file = NumpyHelper.get_mean_per_column_per_df(scores)
            var_over_file = NumpyHelper.get_var_per_column_per_df(scores)

            NumpyHelper.replace_column_with_array(Runtime_Datasets.RUNTIME_MEAN_REPORT, file_index, mean_over_file)
            NumpyHelper.replace_column_with_array(Runtime_Datasets.RUNTIME_VAR_REPORT, file_index, var_over_file)

        if 'memory.max_usage_in_bytes' in df.columns:
            print("Predicting memory...")
            scores = Data_Removal.predict(df, 'memory.max_usage_in_bytes')
            Plotting.tool_evaluation(scores, "memory")
            General_File_Service.create_csv_file(scores, Runtime_Datasets.CURRENT_EVALUATED_TOOL_DIRECTORY, "memory")

            mean_over_file = NumpyHelper.get_mean_per_column_per_df(scores)
            var_over_file = NumpyHelper.get_var_per_column_per_df(scores)

            NumpyHelper.replace_column_with_array(Runtime_Datasets.MEMORY_MEAN_REPORT, file_index, mean_over_file)
            NumpyHelper.replace_column_with_array(Runtime_Datasets.MEMORY_VAR_REPORT, file_index, var_over_file)


    except BaseException as ex:
        print(ex)
        General_File_Service.remove_folder(Runtime_Datasets.CURRENT_WORKING_DIRECTORY)
        sys.exit()


signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    ArgumentParser.handle_args()
    Config.read_conf()
    General_File_Service.check_folder_integrity()
    General_File_Service.create_evaluation_folder()
    General_File_Service.read_files(Config.Config.DATA_RAW_DIRECTORY)
    process_data_sets()
    generate_csv_file()
    plot_data_sets()
    exit(0)
