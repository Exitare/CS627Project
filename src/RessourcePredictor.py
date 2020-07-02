import signal
import sys
from Services.Configuration import Config, Argument_Parser
from Services.FileSystem import Folder_Management
from Services.ToolLoader import Tool_Loader
from RuntimeContants import Runtime_Datasets
import os
import psutil
import logging
import time

logging.basicConfig(filename='example.log', level=logging.DEBUG)
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
    start_time = time.time()
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
    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        tool.evaluate()
        tool.generate_reports()
        tool.generate_plots()
        tool.free_memory()
        logging.info(f"Done.")
        print()

    process = psutil.Process(os.getpid())
    logging.info(f"Memory used: {process.memory_info().rss / 1024 / 1024} mb.")
    end_time = time.time()
    if end_time - start_time > 60:
        logging.info(f"Time passed: {(end_time - start_time) / 60} minutes.")
    else:
        logging.info(f"Time passed: {end_time - start_time} seconds.")
    # Tasks.process_single_files()
    # Tasks.process_merged_tool_version()
    # Tasks.process_single_file_data_removal()
    # Data_Set_Reporting.generate_file_report_files()
    # Add plotting
    logging.info("Done")

    exit(0)
