from Services import NumpyHelper
from Services.Config import Config
import RuntimeContants
from Services.File import General


def write_summary():
    if not NumpyHelper.df_only_nan(RuntimeContants.RUNTIME_MEAN_REPORT):
        RuntimeContants.RUNTIME_MEAN_REPORT['file'] = RuntimeContants.EVALUATED_FILE_NAMES
        RuntimeContants.RUNTIME_MEAN_REPORT['row_count'] = RuntimeContants.EVALUATED_FILE_ROW_COUNTS
        RuntimeContants.RUNTIME_MEAN_REPORT['parameter_count'] = RuntimeContants.EVALUATED_FILE_PARAMETER_COUNTS
        General.create_csv_file(RuntimeContants.RUNTIME_MEAN_REPORT, RuntimeContants.CURRENT_WORKING_DIRECTORY,
                                Config.FILE_RUNTIME_MEAN_SUMMARY)

    if not NumpyHelper.df_only_nan(RuntimeContants.RUNTIME_VAR_REPORT):
        RuntimeContants.RUNTIME_VAR_REPORT['file'] = RuntimeContants.EVALUATED_FILE_NAMES
        RuntimeContants.RUNTIME_VAR_REPORT['row_count'] = RuntimeContants.EVALUATED_FILE_ROW_COUNTS
        RuntimeContants.RUNTIME_VAR_REPORT['parameter_count'] = RuntimeContants.EVALUATED_FILE_PARAMETER_COUNTS
        General.create_csv_file(RuntimeContants.RUNTIME_VAR_REPORT, RuntimeContants.CURRENT_WORKING_DIRECTORY,
                                Config.FILE_RUNTIME_VAR_SUMMARY)

    if not NumpyHelper.df_only_nan(RuntimeContants.MEMORY_MEAN_REPORT):
        RuntimeContants.MEMORY_MEAN_REPORT['file'] = RuntimeContants.EVALUATED_FILE_NAMES
        RuntimeContants.MEMORY_MEAN_REPORT['row_count'] = RuntimeContants.EVALUATED_FILE_ROW_COUNTS
        RuntimeContants.MEMORY_MEAN_REPORT['parameter_count'] = RuntimeContants.EVALUATED_FILE_PARAMETER_COUNTS
        General.create_csv_file(RuntimeContants.MEMORY_MEAN_REPORT, RuntimeContants.CURRENT_WORKING_DIRECTORY,
                                Config.FILE_MEMORY_MEAN_SUMMARY)

    if not NumpyHelper.df_only_nan(RuntimeContants.MEMORY_VAR_REPORT):
        RuntimeContants.MEMORY_VAR_REPORT['file'] = RuntimeContants.EVALUATED_FILE_NAMES
        RuntimeContants.MEMORY_VAR_REPORT['row_count'] = RuntimeContants.EVALUATED_FILE_ROW_COUNTS
        RuntimeContants.MEMORY_VAR_REPORT['parameter_count'] = RuntimeContants.EVALUATED_FILE_PARAMETER_COUNTS
        General.create_csv_file(RuntimeContants.MEMORY_VAR_REPORT, RuntimeContants.CURRENT_WORKING_DIRECTORY,
                                Config.FILE_MEMORY_VAR_SUMMARY)
