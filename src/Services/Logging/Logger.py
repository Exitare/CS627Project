import logging
from enum import Enum
from Services.Configuration.Config import Config

logging.basicConfig(filename='example.log', level=logging.DEBUG)


def setup_logger():
    if Config.DEBUG_MODE:
        logging.basicConfig(filename='example.log', level=logging.DEBUG)
    else:
        logging.basicConfig(filename='example.log', level=logging.INFO)
