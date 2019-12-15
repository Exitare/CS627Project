from sklearn import datasets
from sklearn import __version__
import pandas as pd
import argparse
import copy

from predictions import predict_cpu_usage, predict_memory_usage, predict_total_time
from dataprocessing import convert_factorial_to_numerical, remove_bad_columns, fill_na
import Constants


# https://pbpython.com/categorical-encoding.html

def start():
    """
    Start the application
    :return:
    """
    print(f"Using sklearn version {__version__}")
    # Handle arguments
    args = handle_args()
    Constants.SELECTED_ALGORITHM = args.model
    print(f"Using {Constants.SELECTED_ALGORITHM} model")

    # Load data
    df = load_data(args)
    print("")

    # Processor count prediction
    if 'processor_count' in df.columns:
        print("Predicting cpu count")
        predict_cpu_usage(copy.deepcopy(df))
        print("--------")

    # Total memory usage
    if 'memtotal' in df.columns:
        print("Predicting total memory used")
        predict_memory_usage(copy.deepcopy(df))
        print("--------")

    # Total runtime
    if 'runtime' in df.columns:
        print("Predicting total runtime")
        predict_total_time(df)


def load_data(args):
    """
    Loads the data specific in the args
    :param args:
    :return:
    """
    # load the input file into a pandas dataframe
    if args.filename.endswith(".csv"):
        df = pd.read_csv(args.filename)
    elif args.filename.endswith(".tsv"):
        df = pd.read_csv(args.filename, sep="\t")
    else:
        raise ValueError("unrecognized filetype: %s. I only accept tsv or csv files" % args.filename)

    df = fill_na(df)
    df = remove_bad_columns(df)
    df = convert_factorial_to_numerical(df)
    return df


def handle_args():
    """
    Parse the given arguments
    :return:
    """
    parser = argparse.ArgumentParser(description='Get the impact of tool features on it\'s runtime.',
                                     epilog='Accepts tsv and csv files')
    parser.add_argument('--filename', dest='filename', action='store', required=True)
    parser.add_argument('--model', dest='model', action='store', required=False, default='linear',
                        help='select the desired algorithm. (default: LinearRegression)',
                        choices=[Constants.Model.LASSO.name, Constants.Model.LINEAR.name, Constants.Model.RIDGE.name])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    start()
    exit(0)
