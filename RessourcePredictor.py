from sklearn import __version__
import pandas as pd
import argparse
import numpy as np
from Services.PreProcessing import convert_factorial_to_numerical, remove_bad_columns, fill_na
import Constants
from Services.File import create_file, create_folder
from Services.Plotting import plot_box
from Services.Predictions import predict
import matplotlib.pyplot as plt


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

    folder = create_folder(args)
    if 'runtime' in df.columns:
        print("Predicting runtime...")
        scores = predict(df, 'runtime')
        if folder != "":
            plot_box(scores, folder, "runtime")
            create_file(scores, folder, "runtime")

    print()

    if 'memtotal' in df.columns:
        print("Predicting memory...")
        scores = predict(df, 'memtotal')
        if folder != "":
            plot_box(scores, folder, "memory")
            create_file(scores, folder, "memory")

    elif 'memory.max_usage_in_bytes' in df.columns:
        print("Predicting memory...")
        scores = predict(df, 'memory.max_usage_in_bytes')
        if folder != "":
            plot_box(scores, folder, "memory")
            create_file(scores, folder, "memory")


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
    parser.add_argument('--model', dest='model', action='store', required=False, default='FOREST',
                        help='select the desired algorithm. (default: LinearRegression)',
                        choices=[Constants.Model.LASSO.name, Constants.Model.FOREST.name, Constants.Model.RIDGE.name])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    start()
    exit(0)
