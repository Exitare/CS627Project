import logging
from enum import Enum
from Services.Configuration.Config import Config
import sys

logging.basicConfig(filename='example.log', level=logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
root = logging.getLogger()
root.setLevel(logging.DEBUG)
root.addHandler(handler)