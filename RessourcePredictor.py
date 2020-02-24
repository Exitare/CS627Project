from sklearn import __version__
import pandas as pd
import argparse
import numpy as np
from Services.PreProcessing import convert_factorial_to_numerical, remove_bad_columns, fill_na
import Constants
from Actions import calculate_memory, calculate_runtime
from Services.File import create_file, createFolder
from Services.Plotting import test_plot


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
    print("Predicting...")
    memory_scores = [calculate_memory(df, 0), calculate_memory(df, 10), calculate_memory(df, 20),
                     calculate_memory(df, 30),
                     calculate_memory(df, 40),
                     calculate_memory(df, 50), calculate_memory(df, 60), calculate_memory(df, 70),
                     calculate_memory(df, 80), calculate_memory(df, 90), calculate_memory(df, 99)]

    runtime_scores = [calculate_runtime(df, 0), calculate_runtime(df, 10), calculate_runtime(df, 20),
                      calculate_runtime(df, 30),
                      calculate_runtime(df, 40),
                      calculate_runtime(df, 50), calculate_runtime(df, 60), calculate_runtime(df, 70),
                      calculate_runtime(df, 80), calculate_runtime(df, 90), calculate_runtime(df, 99)]

    folder = createFolder(args)
    if not any(x is None for x in runtime_scores):
        runtimeDF = pd.DataFrame([vars(x) for x in runtime_scores])
        print("Runtime scores:")
        print(runtimeDF)
        if folder != "":
            create_file(runtimeDF, folder, "runtime_scores")

    if not any(x is None for x in memory_scores):
        memoryDF = pd.DataFrame([vars(x) for x in memory_scores])
        print('Memory scores:')
        print(memoryDF)
        if folder != "":
            create_file(memoryDF, folder, "memory_scores")

   # test_plot(len(memoryDF.index), memoryDF.iloc[:, 3])


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
