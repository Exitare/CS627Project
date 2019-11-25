from sklearn import datasets
from sklearn import __version__
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import ast


class Prediction:

    def __init__(self, df):
        self.df = df


def start():
    print(f"Using sklearn version {__version__}")
    handle_args()


def load_data(args):
    # load the input file into a pandas dataframe
    if args.filename.endswith(".csv"):
        prep: Prediction = Prediction(pd.read_csv(args.filename))
    elif args.filename.endswith(".tsv"):
        prep: Prediction = Prediction(pd.read_csv(args.filename, sep="\t"))
    else:
        raise ValueError("unrecognized filetype: %s. I only accept tsv or csv files" % args.filename)

    prep.df = remove_bad_columns(prep.df)
    prep.df.fillna(0, inplace=True)
    array_one = prep.df[0: 9998]
    array_two = prep.df[9998: 19996]
    model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(array_one, array_two, test_size=0.33, random_state=2)
    model.fit(X_train, y_train)
    # print(model.predict(X_test))


def handle_args():
    parser = argparse.ArgumentParser(description='Get the impact of tool features on it\'s runtime.',
                                     epilog='Accepts tsv and csv files')
    parser.add_argument('--filename', dest='filename', action='store', required=True)
    args = parser.parse_args()
    load_data(args)


def remove_bad_columns(df):
    columns = ['ref_file_filetype', 'fastq_input2_filetype', 'fastq_input1_filetype',
               'parameters.analysis_type.algorithmic_options.algorithmic_options_selector',
               'parameters.analysis_type.io_options.io_options_selector',
               'parameters.analysis_type.scoring_options.scoring_options_selector',
               'parameters.fastq_input.fastq_input_selector', 'parameters.rg.rg_selector',
               'parameters.reference_source.index_a', 'parameters.analysis_type.analysis_type_selector',
               'job_runner_name', 'handler', 'destination_id', 'parameters.reference_source.ref_file', 'fastq_input1',
               'fastq_input2']
    for column in columns:
        del df[column]

    return df


if __name__ == '__main__':
    start()
    exit(0)
