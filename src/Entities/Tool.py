import pandas as pd
from pathlib import Path
from Entities.File import File
from RuntimeContants import Runtime_Folders
from Services.FileSystem import Folder_Management


class Tool:
    def __init__(self, name: str):
        self.name = name
        self.combined_data_set = pd.DataFrame()
        # Timestamped folder for this specific run of the application.
        self.evaluation_dir = Runtime_Folders.EVALUATION_DIRECTORY
        # the tool folder
        self.folder = Folder_Management.create_tool_folder(self.name)
        self.files = []

        # if all checks out, the tool will be flag as verified
        # Tools flagged as not verified will not be evaluated
        if self.folder is not None:
            self.verified = True
        else:
            self.verified = False

    def __eq__(self, other):
        return self.name == other.name

    def add_file(self, file_path: str):
        file: File = File(file_path, self.folder)
        self.files.append(file)
