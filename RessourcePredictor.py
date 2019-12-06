from sklearn import datasets
from sklearn import __version__
import pandas as pd
import argparse
import copy

from predictions import predict_cpu_usage, predict_memory_usage
from dataprocessing import convert_factorial_to_numerical


# https://pbpython.com/categorical-encoding.html

def start():
    """
    Start the application
    :return:
    """
    print(f"Using sklearn version {__version__}")
    args = handle_args()
    df = load_data(args)
    predict_cpu_usage(copy.deepcopy(df))
    print("--------")
    predict_memory_usage(copy.deepcopy(df))


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

    df.fillna(0, inplace=True)
    convert_factorial_to_numerical(df)
    return df


def handle_args():
    """
    Parse the given arguments
    :return:
    """
    parser = argparse.ArgumentParser(description='Get the impact of tool features on it\'s runtime.',
                                     epilog='Accepts tsv and csv files')
    parser.add_argument('--filename', dest='filename', action='store', required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    start()
    exit(0)

