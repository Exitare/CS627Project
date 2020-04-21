import configparser
import sys


# Config.py
class Config:
    VERBOSE = 0
    DEBUG = 0
    DATA_ROOT_DIRECTORY = ''
    DATA_RAW_DIRECTORY = ''
    DATA_RESULTS_DIRECTORY = ''
    FILE_RUNTIME_MEAN_SUMMARY = ''
    FILE_RUNTIME_VAR_SUMMARY = ''
    FILE_MEMORY_MEAN_SUMMARY = ''
    FILE_MEMORY_VAR_SUMMARY = ''
    FILE_EXCLUDED_FILES = ''
    K_FOLDS = 0
    REPETITIONS = 0
    FOREST_ESTIMATORS = 0
    MINIMUM_ROW_COUNT = 50


def read_conf():
    """
    Reads the Config.ini File, and stores the values into the Config Class
    :return:
    """
    config = configparser.ConfigParser()
    config.read('src/config.ini')
    try:
        Config.DATA_ROOT_DIRECTORY = config['DATA']['root_directory']
        Config.DATA_RAW_DIRECTORY = config['DATA']['raw_directory']
        Config.DATA_RESULTS_DIRECTORY = config['DATA']['results_directory']

        Config.FILE_RUNTIME_MEAN_SUMMARY = config['FILE_NAMES']['runtime_mean_summary_name']
        Config.FILE_RUNTIME_VAR_SUMMARY = config['FILE_NAMES']['runtime_var_summary_name']
        Config.FILE_MEMORY_MEAN_SUMMARY = config['FILE_NAMES']['memory_mean_summary_name']
        Config.FILE_MEMORY_VAR_SUMMARY = config['FILE_NAMES']['memory_var_summary_name']
        Config.FILE_EXCLUDED_FILES = config['FILE_NAMES']['excluded_files']
        Config.K_FOLDS = int(config['ML']['k_folds'])
        Config.REPETITIONS = int(config['ML']['repetitions'])
        Config.FOREST_ESTIMATORS = int(config['ML']['forest_estimators'])
        Config.MINIMUM_ROW_COUNT = int(config['ML']['min_row_count_per_file'])

        validate_config()
        return True
    except KeyError as ex:
        print(f"Error occurred for key: {ex}")
        print(f"Stopping tool.")
        sys.exit()


def reset_config():
    """
    Resets the Config File. In fact the Config.ini file will be rewritten in total.
    :return:
    """
    config = configparser.ConfigParser()
    config['DATA'] = {
        'data_root_directory': 'Data/',
        'data_raw_directory': 'Data/Raw',
        'data_results_directory': 'Data/Results'
    }
    with open('Config.ini', 'w') as configfile:
        try:
            config.write(configfile)
            configfile.close()
            return True
        except FileNotFoundError as ex:
            return ex


def validate_config():
    print("Validation configuration")
    if not Config.DATA_RESULTS_DIRECTORY:
        print("Please specify a valid Results directory")
        sys.exit()

    if not Config.DATA_RAW_DIRECTORY:
        print("Please specify a valid Raw data directory!")
        sys.exit()

    if not Config.DATA_ROOT_DIRECTORY:
        print("Please specify a valid Root Directory!")
        sys.exit()

    if not Config.FILE_MEMORY_VAR_SUMMARY:
        print(f"Please specify a valid name for FILE_MEMORY_VAR_SUMMARY ")
        sys.exit()

    if not Config.FILE_MEMORY_MEAN_SUMMARY:
        print(f"Please specify a valid name for FILE_MEMORY_MEAN_SUMMARY")
        sys.exit()

    if not Config.FILE_RUNTIME_VAR_SUMMARY:
        print(f"Please specify a valid name for FILE_RUNTIME_VAR_SUMMARY")
        sys.exit()

    if not Config.FILE_RUNTIME_MEAN_SUMMARY:
        print(f"Please specify a valid name for FILE_RUNTIME_MEAN_SUMMARY")
        sys.exit()

    if Config.K_FOLDS <= 3:
        print(f"A negative or value below 3 for folds is invalid. Setting to 5...")
        Config.K_FOLDS = 5
        input()

    if Config.REPETITIONS <= 0:
        print(f"A negative or zero value for repetitions is invalid. Setting to 5...")
        Config.REPETITIONS = 5
        input()

    if Config.REPETITIONS <= 0:
        print(f"A negative or zero value for forest estimators is invalid. Setting to 12...")
        Config.FOREST_ESTIMATORS = 12
        input()

    if Config.MINIMUM_ROW_COUNT < 0:
        print(f"A negative value for the minimum row count is invalid. Setting to 50...")
        Config.MINIMUM_ROW_COUNT = 50
        input()
