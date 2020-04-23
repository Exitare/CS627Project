from RuntimeContants import Runtime_File_Data


def set_general_file_data(filename: str):
    """
    Set default runtime values for each file
    :param filename:
    :return:
    """
    Runtime_File_Data.EVALUATED_FILE_NO_USEFUL_INFORMATION = False
    # Set important runtime file values
    Runtime_File_Data.EVALUATED_FILE_NAME = filename
    # Reduce len of columns by one, because y value is included
    Runtime_File_Data.EVALUATED_FILE_COLUMN_COUNT = len(
        Runtime_File_Data.EVALUATED_FILE_PREPROCESSED_DATA_SET.columns)
    Runtime_File_Data.EVALUATED_FILE_FEATURE_COUNT = Runtime_File_Data.EVALUATED_FILE_COLUMN_COUNT - 1
    Runtime_File_Data.EVALUATED_FILE_ROW_COUNT = len(
        Runtime_File_Data.EVALUATED_FILE_PREPROCESSED_DATA_SET.index)
