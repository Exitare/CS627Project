import pandas as pd
from pathlib import Path
from Entities.File import File
from Services.FileSystem import File_Management
from RuntimeContants import Runtime_Datasets


class Tool:
    def __init__(self, name: str):
        self.name = name
        self.combined_data_set = pd.DataFrame()
        self.folder = Path()
        self.files = []

    def __eq__(self, other):
        return self.name == other.name

    def add_file(self, file_path: str):
        file: File = File(file_path)
        self.files.append(file)

