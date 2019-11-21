
from sklearn import datasets
from sklearn import __version__
import pandas as pd
import argparse

digits = datasets.load_digits()
diabetes = datasets.load_diabetes()


class

def start():
    print(f"Using sklearn version {__version__}")
    handle_args()


def predication(args):
    # load the input file into a pandas dataframe
    if args.filename.endswith(".csv"):
        self.df = pd.read_csv(args.filename)
    elif args.filename.endswith(".tsv"):
        self.df = pd.read_csv(args.filename, sep="\t")
    else:
        raise ValueError("unrecognized filetype: %s. I only accept tsv or csv files" % self.args.filename)


def handle_args():
    parser = argparse.ArgumentParser(description='Get the impact of tool features on it\'s runtime.',
                                     epilog='Accepts tsv and csv files')
    parser.add_argument('--filename', dest='filename', action='store', required=True)
    parser.add_argument("--runtime_label", dest='runtime_label', action='store', default="runtime")
    parser.add_argument("--split_train_test", dest='split_train_test', action='store', default="False")
    parser.add_argument("--split_randomly", dest='split_randomly', action='store', default="True")
    parser.add_argument('--plot_outfile', dest='plot_outfile', action='store', default="plot.png",
                        help='png output file.')
    parser.add_argument('--model_outfile', dest='model_outfile', action='store', default='model.pkl',
                        help='pkl output file.')
    args = parser.parse_args()
    predication(args)


if __name__ == '__main__':
    start()
    exit(0)
