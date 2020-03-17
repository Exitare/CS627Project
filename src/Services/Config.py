import configparser
import sys


# Config.py
class Config:
    VERBOSE = 0
    DEBUG = 0
    DATA_ROOT_DIRECTORY = ''
    DATA_RAW_DIRECTORY = ''
    DATA_RESULTS_DIRECTORY = ''


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
