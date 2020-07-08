import logging
import os
import time
import psutil
from RuntimeContants import Misc


def get_application_stats():
    """
    Calculates statistic data for a run of the application
    """
    process = psutil.Process(os.getpid())
    logging.info(f"Memory used: {process.memory_info().rss / 1024 / 1024} mb.")
    end_time = time.time()
    if end_time - Misc.application_start_time > 60:
        logging.info(f"Time passed: {(end_time - Misc.application_start_time) / 60} minutes.")
    else:
        logging.info(f"Time passed: {end_time - Misc.application_start_time} seconds.")


def get_duration(start_time: float):
    """
    Calculates the duration between a start and end point.
    """
    end_time = time.time()
    return end_time - start_time
