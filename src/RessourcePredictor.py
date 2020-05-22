import signal
import sys
from Services import Config
from Services.FileSystem import FileManagement, FolderManagement
from RuntimeContants import Runtime_Folders
from Services import ArgumentParser
from src import Tasks
from Services.Reporting import Data_Set_Reporting, Plots

def signal_handler(sig, frame):
    """
    Handles a signal. Like pressing crtl +c
    :param sig:
    :param frame:
    :return:
    """
    print('Shutting down gracefully!')
    print("Deleting working directory")
    FolderManagement.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
    print("Done")
    print("Bye")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    ArgumentParser.handle_args()
    Config.read_conf()
    FolderManagement.initialize()
    FileManagement.load_required_data()
    Tasks.process_single_files()
    Tasks.process_merged_tool_version()
    Tasks.process_single_file_data_removal()
    Data_Set_Reporting.generate_file_report_files()
    # Add plotting
    print("Done")
    exit(0)
