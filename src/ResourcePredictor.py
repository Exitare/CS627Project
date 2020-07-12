import signal
import sys
from Services.Configuration import Config, Argument_Parser
from Services.FileSystem import Folder_Management
from Services.ToolLoader import Tool_Loader
from RuntimeContants import Runtime_Datasets
from Services.Statistics import Runtime_Statistics
import logging
import time

logging.basicConfig(filename='log.log', level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
root = logging.getLogger()
root.setLevel(logging.DEBUG)
root.addHandler(handler)


def signal_handler(sig, frame):
    """
    Handles a signal. Like pressing crtl +c
    :param sig:
    :param frame:
    :return:
    """
    print('Shutting down gracefully!')
    print("Done")
    print("Bye")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    Runtime_Statistics.application_start_time = time.time()
    Config.read_conf()
    Argument_Parser.handle_args()

    if Folder_Management.create_required_folders():
        logging.info("All required folders generated.")
        logging.info("Please copy your files into the new folders and restart the application.")
        exit(0)
    else:
        logging.info("All folder checks passed.")
        logging.info("Creating evaluation folder.")
        Folder_Management.create_evaluation_folder()

    Tool_Loader.load_tools()

    logging.info("Starting tool evaluation...")
    print()
    # Tool evaluation workflow
    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        tool_start_time = time.time()
        tool.evaluate()
        tool.generate_reports()
        tool.generate_plots()
        tool.free_memory()

        time_passed = Runtime_Statistics.get_duration(tool_start_time)
        print()
        if time_passed > 60:
            logging.info(f"Tool {tool.name} evaluated in {time_passed} minutes")
        else:
            logging.info(f"Tool {tool.name} evaluated in {time_passed} seconds")
        print()

    # TODO: Add best/worst performing tools

    Runtime_Statistics.get_application_stats()

    logging.info("Done")

    exit(0)
