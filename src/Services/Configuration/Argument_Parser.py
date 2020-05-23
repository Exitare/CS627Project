from RuntimeContants import Runtime_Datasets
import argparse

COMMAND_LINE_ARGS = []

remove = False
verbose = False


def handle_args():
    """
    Parse the given arguments
    :return:
    """
    global COMMAND_LINE_ARGS
    global remove
    global verbose

    parser = argparse.ArgumentParser(description='Get the impact of tool features on it\'s runtime.',
                                     epilog='Accepts tsv and csv files')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store', required=False,
                        help="Enables the verbose mode. With active verbose mode additional information is shown in the console")
    parser.add_argument('-m', '--merge', dest='merge', default=False, required=False,
                        help="Enable the detection and evaluation of multiple versions of tools present in the raw folder")
    parser.add_argument('-r', '--remove', dest='remove', action='store', required=False,
                        help="Activates the removal of data for further evaluation of data sets")
    args = parser.parse_args()
    COMMAND_LINE_ARGS = args

    if COMMAND_LINE_ARGS.remove:
        print("Data removal active")
        remove = True

    if COMMAND_LINE_ARGS.verbose:
        print("Verbose output active")
        verbose = True
