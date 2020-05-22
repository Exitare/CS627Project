

def generate_file_report_files():
    # Write general information about the data set
    General_File_Service.create_csv_file(Runtime_File_Data.EVALUATED_FILE_RUNTIME_INFORMATION,
                                         Runtime_Folders.CURRENT_WORKING_DIRECTORY,
                                         "General_Information_Runtime")
    General_File_Service.create_csv_file(Runtime_File_Data.EVALUATED_FILE_MEMORY_INFORMATION,
                                         Runtime_Folders.CURRENT_WORKING_DIRECTORY,
                                         "General_Information_Memory")
    # Write general information for a specific tool , non modified
    General_File_Service.create_csv_file(Runtime_File_Data.EVALUATED_FILE_REMOVED_ROWS_RUNTIME_INFORMATION,
                                         Runtime_Folders.CURRENT_EVALUATED_TOOL_DIRECTORY,
                                         "data_removal_runtime_evaluation")
    General_File_Service.create_csv_file(Runtime_File_Data.EVALUATED_FILE_REMOVED_ROWS_MEMORY_INFORMATION,
                                         Runtime_Folders.CURRENT_EVALUATED_TOOL_DIRECTORY,
                                         "data_removal_memory_evaluation")


def generate_generate_report_files():
    """
    Writes all specified data sets
    :return:
    """

    # Write the mean and var reports for all files
    General_File_Service.create_csv_file(Runtime_Datasets.RUNTIME_VAR_REPORT,
                                         Runtime_Folders.CURRENT_WORKING_DIRECTORY,
                                         Config.Config.FILE_RUNTIME_VAR_SUMMARY)

    General_File_Service.create_csv_file(Runtime_Datasets.RUNTIME_MEAN_REPORT,
                                         Runtime_Folders.CURRENT_WORKING_DIRECTORY,
                                         Config.Config.FILE_RUNTIME_MEAN_SUMMARY)

    General_File_Service.create_csv_file(Runtime_Datasets.MEMORY_MEAN_REPORT,
                                         Runtime_Folders.CURRENT_WORKING_DIRECTORY,
                                         Config.Config.FILE_MEMORY_MEAN_SUMMARY)

    General_File_Service.create_csv_file(Runtime_Datasets.MEMORY_VAR_REPORT,
                                         Runtime_Folders.CURRENT_WORKING_DIRECTORY,
                                         Config.Config.FILE_MEMORY_VAR_SUMMARY)

    General_File_Service.create_csv_file(Runtime_Datasets.EXCLUDED_FILES, Runtime_Folders.CURRENT_WORKING_DIRECTORY,
                                         Config.Config.FILE_EXCLUDED_FILES)
