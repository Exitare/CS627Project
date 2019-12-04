from sklearn import datasets
from sklearn import __version__
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import copy
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate


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
    decode_data(df)
    return df


def hyper_parameter(X, y):
    ridge = make_pipeline(LinearRegression())
    print(ridge.get_params())

    hyperparameters = [{'linearfeatures__degree': range(1, 10)}]
    # the keywords have to match keys in pipe.get_params().keys()

    hyp = GridSearchCV(ridge, hyperparameters,
                       cv=StratifiedKFold(3, shuffle=True),
                       n_jobs=-1)  # parallelize

    result = cross_validate(hyp, X, y,
                            cv=StratifiedKFold(4, shuffle=True),
                            return_estimator=True)

    print(result['test_score'], [est.best_params_ for est in result['estimator']])


def predict_cpu_usage(df):
    """
    Trains the model to predict the cpu usage
    :param df:
    :return:
    """
    print("CPU usage predication started...")
    data = remove_bad_columns(df)

    y = data['processor_count']
    del data['processor_count']
    X = data

    print("Training model...")

    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.33, random_state=2)
    model.fit(X_train, y_train)
    y_test_hat = model.predict(X_train)

    print(f"CPU model test score is : {model.score(X_test, y_test)}")
    print(f"CPU model train score is : {model.score(X_train, y_train)}")
    print(f"Predication: {y_test_hat[:5]}")
    print(f"Y Values: {y_train[:5]}")

    scores = cross_val_score(model, X, y, cv=5)
    print(f"CPU Cross validation score is : {scores}")
    show_plot(X_train, y_train, "square")


# hyper_parameter(X, y)
# Lasso, Ridge
# Grid search
# train data split again, validation data using to set hyperparamter

def predict_memory_usage(df):
    """
    Trains the model to predict memory usage
    :param df:
    :return:
    """
    print("Memory usage predication started...")
    data = remove_bad_columns(df)

    y = data['memtotal']
    del data['memtotal']
    X = data

    model = LinearRegression()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
    model.fit(X_train, y_train)
    y_test_hat = model.predict(X_train)
    print(f"Memory model test score is : {model.score(X_test, y_test)}")
    print(f"Memory model train score is : {model.score(X_train, y_train)}")
    print(f"Predication: {y_test_hat}")

    scores = cross_val_score(model, X, y, cv=5)
    print(f"Memory Cross validation score is : {scores}")
    # show_plot(y_test, y_test_hat, "square")


def decode_data(df):
    """
    De
    :param df:
    :return:
    """
    print("Decoding categoriacal data...")
    cleanup_nums = {"fastq_input2_filetype": {"none": 0, "uncompressed": 1, "compressed": 2},
                    "fastq_input1_filetype": {"compressed": 0, "uncompressed": 1}}
    df.replace(cleanup_nums, inplace=True)


def show_plot(x, y, axis: str):
    plt.plot([0, 2.5], [0, 2.5], 'k', lw=0.5)  # reference diagonal
    plt.plot(x, y, '.')
    plt.axis(axis)
    plt.show()


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


def remove_bad_columns(dataframe):
    print("Removing columns...")
    columns = ['ref_file_filetype',
               'parameters.analysis_type.algorithmic_options.algorithmic_options_selector',
               'parameters.analysis_type.io_options.io_options_selector',
               'parameters.analysis_type.scoring_options.scoring_options_selector',
               'parameters.fastq_input.fastq_input_selector', 'parameters.rg.rg_selector',
               'parameters.reference_source.index_a', 'parameters.analysis_type.analysis_type_selector',
               'job_runner_name', 'handler', 'destination_id', 'parameters.reference_source.ref_file']

    for column in columns:
        del dataframe[column]
    # use pd to remove
    return dataframe


if __name__ == '__main__':
    start()
    exit(0)
