from sklearn import __version__
import pandas as pd
import argparse
import signal
import sys
from Services import  NumpyHelper, Config, Plotting
from Services.Predictions import Single_Predictions, Data_Removal
from Services.File import General, Data_Removal
import RuntimeContants
import numpy as np


# https://pbpython.com/categorical-encoding.html

def start():
    """
    Start the application
    :return:
    """
    Config.read_conf()
    General.check_folder_integrity()
    General.create_evaluation_folder()
    data_frames = General.read_files(Config.Config.DATA_RAW_DIRECTORY)

    RuntimeContants.RUNTIME_MEAN_REPORT = pd.DataFrame(np.nan, index=np.arange(len(data_frames)),
                                                       columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90',
                                                          '91', '92', '93', '94', '95', '96', '97', '98', '99'])
    RuntimeContants.RUNTIME_VAR_REPORT = pd.DataFrame(np.nan, index=np.arange(len(data_frames)),
                                                      columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90',
                                                         '91', '92', '93', '94', '95', '96', '97', '98', '99'])

    RuntimeContants.MEMORY_MEAN_REPORT = pd.DataFrame(np.nan, index=np.arange(len(data_frames)),
                                                      columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90',
                                                         '91', '92', '93', '94', '95', '96', '97', '98', '99'])
    RuntimeContants.MEMORY_VAR_REPORT = pd.DataFrame(np.nan, index=np.arange(len(data_frames)),
                                                     columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90',
                                                        '91', '92', '93', '94', '95', '96', '97', '98', '99'])

    if len(data_frames) == 0:
        print("No files found to evaluate. Stopping")
        General.remove_folder(RuntimeContants.CURRENT_WORKING_DIRECTORY)
        sys.exit()

    file_index = 0
    for filename, df in data_frames.items():
        try:
            RuntimeContants.CURRENT_EVALUATED_FILE = filename
            RuntimeContants.EVALUATED_FILE_NAMES.append(filename)
            RuntimeContants.EVALUATED_FILE_PARAMETER_COUNTS.append(len(df.columns))
            RuntimeContants.EVALUATED_FILE_ROW_COUNTS.append(len(df.index))
            General.create_tool_folder(filename)

            print(1)
            # Working on full data set
            Single_Predictions.compare_real_to_predicted_data(df)
            print(2)
            if RuntimeContants.COMMAND_LINE_ARGS.remove:
                remove_data(filename, df, file_index)

        except Exception as ex:
            print("error occured in start()")
            print(ex)
            General.remove_folder(RuntimeContants.CURRENT_WORKING_DIRECTORY)
            sys.exit()

        # Increase file index to replace the correct row in the previous made data set
        file_index += 1


    General.create_csv_file(RuntimeContants.OVER_UNDER_FITTING, RuntimeContants.CURRENT_WORKING_DIRECTORY, "Over_Under_Fitting")

    if RuntimeContants.COMMAND_LINE_ARGS.remove:
        Plotting.plot_summary()
        Data_Removal.write_summary()
        Plotting.plot_group_by_parameter_count()


def handle_args():
    """
    Parse the given arguments
    :return:
    """
    parser = argparse.ArgumentParser(description='Get the impact of tool features on it\'s runtime.',
                                     epilog='Accepts tsv and csv files')
    parser.add_argument('--path', dest='path', action='store', required=False, choices=[0, 1], default=0)
    parser.add_argument('--remove', dest='remove', action='store', required=False)
    args = parser.parse_args()
    RuntimeContants.COMMAND_LINE_ARGS = args


def signal_handler(sig, frame):
    print('Shutting down gracefully!')
    print("Deleting working directory")
    General.remove_folder(RuntimeContants.CURRENT_WORKING_DIRECTORY)
    print("Done")
    print("Bye")
    sys.exit(0)


def remove_data(filename: str, df, file_index: int):
    try:
        print(f"Evaluating {filename}")
        if 'runtime' in df.columns:
            print("Predicting runtime...")
            scores = Data_Removal.predict(df, 'runtime')
            Plotting.tool_evaluation(scores, "runtime")
            General.create_csv_file(scores, RuntimeContants.CURRENT_EVALUATED_TOOL_DIRECTORY, "runtime")

            mean_over_file = NumpyHelper.get_mean_per_column_per_df(scores)
            var_over_file = NumpyHelper.get_var_per_column_per_df(scores)
            NumpyHelper.replace_column_with_array(RuntimeContants.RUNTIME_MEAN_REPORT, file_index, mean_over_file)
            NumpyHelper.replace_column_with_array(RuntimeContants.RUNTIME_VAR_REPORT, file_index, var_over_file)

        if 'memory.max_usage_in_bytes' in df.columns:
            print("Predicting memory...")
            scores = Data_Removal.predict(df, 'memory.max_usage_in_bytes')
            Plotting.tool_evaluation(scores, "memory")
            General.create_csv_file(scores, RuntimeContants.CURRENT_EVALUATED_TOOL_DIRECTORY, "memory")

            mean_over_file = NumpyHelper.get_mean_per_column_per_df(scores)
            var_over_file = NumpyHelper.get_var_per_column_per_df(scores)
            NumpyHelper.replace_column_with_array(RuntimeContants.MEMORY_MEAN_REPORT, file_index, mean_over_file)
            NumpyHelper.replace_column_with_array(RuntimeContants.MEMORY_VAR_REPORT, file_index, var_over_file)


    except BaseException as ex:
        print(ex)
        General.remove_folder(RuntimeContants.CURRENT_WORKING_DIRECTORY)
        sys.exit()


signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    handle_args()
    start()
    exit(0)
