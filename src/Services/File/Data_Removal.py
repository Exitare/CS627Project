
from Services.Config import Config
import RuntimeContants
from Services.File import General_File_Service
from RuntimeContants import Runtime_Datasets, Runtime_Folders


def write_summary():
    if not Runtime_Datasets.RUNTIME_MEAN_REPORT.empty:
        Runtime_Datasets.RUNTIME_MEAN_REPORT['file'] = Runtime_Datasets.EVALUATED_FILE_NAMES
        Runtime_Datasets.RUNTIME_MEAN_REPORT['row_count'] = Runtime_Datasets.EVALUATED_FILE_ROW_COUNTS
        Runtime_Datasets.RUNTIME_MEAN_REPORT['parameter_count'] = Runtime_Datasets.EVALUATED_FILE_PARAMETER_COUNTS
        General_File_Service.create_csv_file(Runtime_Datasets.RUNTIME_MEAN_REPORT,
                                             Runtime_Datasets.CURRENT_WORKING_DIRECTORY,
                                             Config.FILE_RUNTIME_MEAN_SUMMARY)

    if not Runtime_Datasets.RUNTIME_VAR_REPORT.empty:
        Runtime_Datasets.RUNTIME_VAR_REPORT['file'] = Runtime_Datasets.EVALUATED_FILE_NAMES
        Runtime_Datasets.RUNTIME_VAR_REPORT['row_count'] = Runtime_Datasets.EVALUATED_FILE_ROW_COUNTS
        Runtime_Datasets.RUNTIME_VAR_REPORT['parameter_count'] = Runtime_Datasets.EVALUATED_FILE_PARAMETER_COUNTS
        General_File_Service.create_csv_file(Runtime_Datasets.RUNTIME_VAR_REPORT,
                                             Runtime_Folders.CURRENT_WORKING_DIRECTORY,
                                             Config.FILE_RUNTIME_VAR_SUMMARY)

    if not Runtime_Datasets.MEMORY_MEAN_REPORT.empty:
        RuntimeContants.Runtime_Datasets['file'] = Runtime_Datasets.EVALUATED_FILE_NAMES
        RuntimeContants.Runtime_Datasets['row_count'] = Runtime_Datasets.EVALUATED_FILE_ROW_COUNTS
        RuntimeContants.Runtime_Datasets['parameter_count'] = Runtime_Datasets.EVALUATED_FILE_PARAMETER_COUNTS
        General_File_Service.create_csv_file(Runtime_Datasets.MEMORY_MEAN_REPORT,
                                             Runtime_Folders.CURRENT_WORKING_DIRECTORY,
                                             Config.FILE_MEMORY_MEAN_SUMMARY)

    if not Runtime_Datasets.MEMORY_VAR_REPORT.empty:
        RuntimeContants.Runtime_Datasets['file'] = Runtime_Datasets.EVALUATED_FILE_NAMES
        RuntimeContants.Runtime_Datasets['row_count'] = Runtime_Datasets.EVALUATED_FILE_ROW_COUNTS
        RuntimeContants.Runtime_Datasets['parameter_count'] = Runtime_Datasets.EVALUATED_FILE_PARAMETER_COUNTS
        General_File_Service.create_csv_file(Runtime_Datasets.MEMORY_VAR_REPORT,
                                             Runtime_Folders.CURRENT_WORKING_DIRECTORY,
                                             Config.FILE_MEMORY_VAR_SUMMARY)
