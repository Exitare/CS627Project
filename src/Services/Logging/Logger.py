import logging
from enum import Enum

logging.basicConfig(filename='example.log', level=logging.DEBUG)


class LogLevel(Enum):


def write_log(LogLevel: LogLevel):
