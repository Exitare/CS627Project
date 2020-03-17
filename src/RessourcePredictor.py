from sklearn import __version__
import pandas as pd
import argparse
import signal
import sys
from Services import File, Predictions
from Services.Plotting import plot_box
from Services import Config
import Constants


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
    print(data_frames)
    for filename, df in data_frames.items():
        print(f"Evaluation {filename}")
        File.create_tool_folder(filename)
        if 'runtime' in df.columns:
            print("Predicting runtime...")
            scores = Predictions.predict(df, 'runtime')
            plot_box(scores, Constants.CURRENT_EVALUATED_TOOL_DIRECTORY, "runtime")
            File.create_csv_file(scores, Constants.CURRENT_EVALUATED_TOOL_DIRECTORY, "runtime")

        print()

        if 'memory.max_usage_in_bytes' in df.columns:
            print("Predicting memory...")
            scores = Predictions.predict(df, 'memory.max_usage_in_bytes')
            plot_box(scores, Constants.CURRENT_EVALUATED_TOOL_DIRECTORY, "memory")
            File.create_csv_file(scores, Constants.CURRENT_EVALUATED_TOOL_DIRECTORY, "memory")


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
