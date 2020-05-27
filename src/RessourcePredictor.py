import signal
import sys
from Services.Configuration import Config, Argument_Parser
from Services.FileSystem import File_Management, Folder_Management
from RuntimeContants import Runtime_Folders
import os
import psutil


def signal_handler(sig, frame):
    """
    Handles a signal. Like pressing crtl +c
    :param sig:
    :param frame:
    :return:
    """
    print('Shutting down gracefully!')
    print("Deleting working directory")
    Folder_Management.remove_folder(Runtime_Folders.EVALUATION_DIRECTORY)
    print("Done")
    print("Bye")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    Config.read_conf()
    Argument_Parser.handle_args()
    Folder_Management.initialize()
    File_Management.load_tools()

    process = psutil.Process(os.getpid())
    print(f"Memory used: {process.memory_info().rss / 1024} mb.")
    # Tasks.process_single_files()
    # Tasks.process_merged_tool_version()
    # Tasks.process_single_file_data_removal()
    # Data_Set_Reporting.generate_file_report_files()
    # Add plotting
    print("Done")
    exit(0)
