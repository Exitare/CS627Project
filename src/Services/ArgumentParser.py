from RuntimeContants import Runtime_Datasets
import argparse


def handle_args():
    """
    Parse the given arguments
    :return:
    """
    parser = argparse.ArgumentParser(description='Get the impact of tool features on it\'s runtime.',
                                     epilog='Accepts tsv and csv files')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store', required=False)
    parser.add_argument('-r', '--remove', dest='remove', action='store', required=False)
    args = parser.parse_args()
    Runtime_Datasets.COMMAND_LINE_ARGS = args

    if Runtime_Datasets.COMMAND_LINE_ARGS.remove:
        print("Data removal active")

    if Runtime_Datasets.COMMAND_LINE_ARGS.verbose:
        print("Verbose output active")

