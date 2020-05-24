import pandas as pd
from collections import defaultdict

# All data sets from the folder
RAW_FILE_PATHS = []
RAW_SIMILAR_FILES = defaultdict(list)

# Data Removal
RUNTIME_MEAN_REPORT = pd.DataFrame(columns=['File', '0', '10', '20', '30', '40', '50', '60', '70', '80',
                                            '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', 'Rows',
                                            'Features'])
RUNTIME_VAR_REPORT = pd.DataFrame(columns=['File', '0', '10', '20', '30', '40', '50', '60', '70', '80',
                                           '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', 'Rows',
                                           'Features'])
MEMORY_MEAN_REPORT = pd.DataFrame(columns=['File', '0', '10', '20', '30', '40', '50', '60', '70', '80',
                                           '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', 'Rows',
                                           'Features'])
MEMORY_VAR_REPORT = pd.DataFrame(columns=['File', '0', '10', '20', '30', '40', '50', '60', '70', '80',
                                          '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', 'Rows',
                                          'Features'])
K_FOLD_FINAL_SCORES = pd.DataFrame(columns=['File', '0', '10', '20', '30', '40', '50', '60', '70', '80',
                                            '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', 'Rows',
                                            'Features'])

EXCLUDED_FILES = pd.DataFrame(columns=['File', 'Rows'])

EVALUATED_FILE_NAMES = []
# Backwards compatibility
EVALUATED_FILES_ROW_COUNTS = []
EVALUATED_FILES_PARAMETER_COUNTS = []


FILES = []
