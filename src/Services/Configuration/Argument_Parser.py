from RuntimeContants import Runtime_Datasets
import argparse
from Services.Configuration.Config import Config
from time import sleep


def handle_args():
    """
    Parse the given arguments
    :return:
    """

    parser = argparse.ArgumentParser(description='Get the impact of tool features on it\'s runtime.',
                                     epilog='Accepts tsv and csv files')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store', required=False,
                        help="Enables the verbose mode. With active verbose mode additional information is shown in the console")
    parser.add_argument('-p', '--percentage', dest='percentage', default=False, required=False,
                        help="Enable the detection and evaluation of multiple versions of tools present in the raw folder")
    parser.add_argument('-r', '--remove', dest='remove', action='store', required=False,
                        help="Activates the removal of data for further evaluation of data sets")
    parser.add_argument('-m', '--memory', dest='memory', action='store', required=False,
                        help="If set, the application will run in memory saving mode. Should only be used where memory is limited.")
    parser.add_argument('-d', '--debug', dest='debug', action='store', required=False,
                        help="If set, the tool will run in debug mode. You will get developer output. The performance"
                             "is most likely be not as fast as possible!")
    args = parser.parse_args()

    if args.remove:
        print("Data removal active")
        Config.PERCENTAGE_REMOVAL = True

    if args.verbose:
        print("Verbose output active")
        Config.VERBOSE = True

    if args.percentage:
        print("Tool files will be merged and then evaluated")
        Config.MERGED_TOOL_EVALUATION = True

    if args.memory:
        print(f"Memory saving mode is active!")
        Config.MEMORY_SAVING_MODE = True

    if args.debug:
        print(f"Debug mode is active!")
        Config.DEBUG_MODE = True

    sleep(1)
