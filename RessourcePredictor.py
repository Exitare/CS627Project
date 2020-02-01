from sklearn import datasets
from sklearn import __version__
import pandas as pd
import numpy as np
import argparse
import copy

from predictions import predict_cpu_usage, predict_memory_usage, predict_total_time
from dataprocessing import convert_factorial_to_numerical, remove_bad_columns, fill_na, remove_random_rows
import Constants
import stats
from Score import Score


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

    memoryScores = Score({"value": 0, "rowcount": 0}, {"value": 0, "rowcount": 0}, {"value": 0, "rowcount": 0})
    runtimeScores = Score({"value": 0, "rowcount": 0}, {"value": 0, "rowcount": 0}, {"value": 0, "rowcount": 0})

    df = load_data(args)
    deepCopy = df
    for i in range(len(deepCopy.index)):
        print("")
        df = deepCopy
        df = remove_random_rows(df, i)

        print("-------------")
        print(i)
        print(f"Test with {len(df.index)} rows")
        print("")
        # Total memory usage
        if 'memtotal' in df.columns:
            model, testScore, trainScore, crossScore = predict_memory_usage(copy.deepcopy(df))

            crossScore = np.mean(crossScore)

            if float(testScore) >= memoryScores.testScore.get("value"):
                memoryScores.testScore = {"value": testScore, "rowcount": len(df.index)}

            if float(trainScore) >= memoryScores.trainScore.get("value"):
                memoryScores.trainScore = {"value": trainScore, "rowcount": len(df.index)}

            if float(crossScore) >= memoryScores.crossValidationScore.get("value"):
                memoryScores.crossValidationScore = {"value": crossScore, "rowcount": len(df.index)}

        #  if Constants.SELECTED_ALGORITHM != Constants.Model.FOREST.name:
        #     stats.print_coef(df, model)

        # If model selection is Ridge print best alpha value
        # if Constants.SELECTED_ALGORITHM == Constants.Model.RIDGE.name:
        #   stats.print_alpha(model)

        # Total runtime
        if 'runtime' in df.columns:
            print("Predicting total runtime")
            model, testScore, trainScore, crossScore = predict_total_time(df)
            crossScore = np.mean(crossScore)

            if float(testScore) >= runtimeScores.testScore.get("value"):
                runtimeScores.testScore = {"value": testScore, "rowcount": len(df.index)}

            if float(trainScore) >= runtimeScores.trainScore.get("value"):
                runtimeScores.trainScore = {"value": trainScore, "rowcount": len(df.index)}

            if float(crossScore) >= runtimeScores.crossValidationScore.get("value"):
                runtimeScores.crossValidationScore = {"value": crossScore, "rowcount": len(df.index)}

        # if Constants.SELECTED_ALGORITHM != Constants.Model.FOREST.name:
        #    stats.print_coef(df, model)

        # If model selection is Ridge print best alpha value
        # if Constants.SELECTED_ALGORITHM == Constants.Model.RIDGE.name:
        #   stats.print_alpha(model)


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
