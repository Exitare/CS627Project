from sklearn import __version__
import pandas as pd
import argparse
import signal
import sys
from Services import File, Predictions, NumpyHelper, Config, Plotting
import Constants
import numpy as np


# https://pbpython.com/categorical-encoding.html

def start():
    """
    Start the application
    :return:
    """
    Config.read_conf()
    File.check_folder_integrity()
    File.create_evaluation_folder()
    data_frames = File.read_files(Config.Config.DATA_RAW_DIRECTORY)

    Constants.RUNTIME_MEAN_REPORT = pd.DataFrame(np.nan, index=np.arange(len(data_frames)),
                                                 columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90',
                                                          '91', '92', '93', '94', '95', '96', '97', '98', '99'])
    Constants.RUNTIME_VAR_REPORT = pd.DataFrame(np.nan, index=np.arange(len(data_frames)),
                                                columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90',
                                                         '91', '92', '93', '94', '95', '96', '97', '98', '99'])

    Constants.MEMORY_MEAN_REPORT = pd.DataFrame(np.nan, index=np.arange(len(data_frames)),
                                                columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90',
                                                         '91', '92', '93', '94', '95', '96', '97', '98', '99'])
    Constants.MEMORY_VAR_REPORT = pd.DataFrame(np.nan, index=np.arange(len(data_frames)),
                                               columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90',
                                                        '91', '92', '93', '94', '95', '96', '97', '98', '99'])

    if len(data_frames) == 0:
        print("No files found to evaluate. Stopping")
        File.remove_folder(Constants.CURRENT_WORKING_DIRECTORY)
        sys.exit()

    file_index = 0
    for filename, df in data_frames.items():
        try:
            Constants.EVALUATED_FILE_NAMES.append(filename)

            print(f"Evaluating {filename}")
            File.create_tool_folder(filename)
            if 'runtime' in df.columns:
                print("Predicting runtime...")
                scores = Predictions.predict(df, 'runtime')
                Plotting.plot_box(scores, Constants.CURRENT_EVALUATED_TOOL_DIRECTORY, "runtime")
                File.create_csv_file(scores, Constants.CURRENT_EVALUATED_TOOL_DIRECTORY, "runtime")

                mean_over_file = NumpyHelper.get_mean_per_column_per_df(scores)
                var_over_file = NumpyHelper.get_var_per_column_per_df(scores)
                NumpyHelper.replace_column_with_array(Constants.RUNTIME_MEAN_REPORT, file_index, mean_over_file)
                NumpyHelper.replace_column_with_array(Constants.RUNTIME_VAR_REPORT, file_index, var_over_file)

            if 'memory.max_usage_in_bytes' in df.columns:
                print("Predicting memory...")
                scores = Predictions.predict(df, 'memory.max_usage_in_bytes')
                Plotting.plot_box(scores, Constants.CURRENT_EVALUATED_TOOL_DIRECTORY, "memory")
                File.create_csv_file(scores, Constants.CURRENT_EVALUATED_TOOL_DIRECTORY, "memory")

                mean_over_file = NumpyHelper.get_mean_per_column_per_df(scores)
                var_over_file = NumpyHelper.get_var_per_column_per_df(scores)
                NumpyHelper.replace_column_with_array(Constants.MEMORY_MEAN_REPORT, file_index, mean_over_file)
                NumpyHelper.replace_column_with_array(Constants.MEMORY_VAR_REPORT, file_index, var_over_file)

        except Exception as ex:
            print(ex)
            File.remove_folder(Constants.CURRENT_WORKING_DIRECTORY)
            sys.exit()

        # Increase file index to replace the correct row in the previous made data set
        file_index += 1

    File.write_summary()


def handle_args():
    """
    Parse the given arguments
    :return:
    """
    parser = argparse.ArgumentParser(description='Get the impact of tool features on it\'s runtime.',
                                     epilog='Accepts tsv and csv files')
    parser.add_argument('--path', dest='path', action='store', required=False)
    args = parser.parse_args()
    return args


def signal_handler(sig, frame):
    print('Shutting down gracefully!')
    print("Deleting working directory")
    File.remove_folder(Constants.CURRENT_WORKING_DIRECTORY)
    print("Done")
    print("Bye")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    start()
    exit(0)
