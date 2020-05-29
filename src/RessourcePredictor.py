import signal
import sys
from Services.Configuration import Config, Argument_Parser
from Services.FileSystem import Folder_Management
from Services.ToolLoader import Tool_Loader
from RuntimeContants import Runtime_Folders, Runtime_Datasets
import os
import psutil
from Services.Logging import Logger
import logging


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
    Config.read_conf()
    Argument_Parser.handle_args()

    if Folder_Management.create_required_folders():
        logging.info("All required folders generated.")
        logging.info("Please copy your files into the new folders and restart the application.")
        exit(0)

    Tool_Loader.load_tools()

    # for tool in Runtime_Datasets.VERIFIED_TOOLS:
    #   logging.info(f"Evaluation tool {tool.name}...")
    #   tool.evaluate_files()
    #   logging.info(f"Done.")
    #   print()

    process = psutil.Process(os.getpid())
    print(f"Memory used: {process.memory_info().rss / 1024} mb.")
    # Tasks.process_single_files()
    # Tasks.process_merged_tool_version()
    # Tasks.process_single_file_data_removal()
    # Data_Set_Reporting.generate_file_report_files()
    # Add plotting
    print("Done")
    exit(0)
