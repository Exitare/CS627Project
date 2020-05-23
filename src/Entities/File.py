from pathlib import Path

class File:
    def __init__(self, name: str, path: Path(), ):
        self.testScore = testScore
        self.trainScore = trainScore
        self.crossValidationScore = crossValidationScore
        self.variance = variance


EVALUATED_FILE_ROW_COUNT = 0
EVALUATED_FILE_COLUMN_COUNT = 0
EVALUATED_FILE_FEATURE_COUNT = 0
EVALUATED_FILE_RAW_DATA_SET = pd.DataFrame()
EVALUATED_FILE_PREPROCESSED_DATA_SET = pd.DataFrame()
EVALUATED_FILE_NO_USEFUL_INFORMATION = False

# General information about a data set with removed rows
EVALUATED_FILE_REMOVED_ROWS_RUNTIME_INFORMATION = pd.DataFrame(
    columns=['0', '10', '20', '30', '40', '50', '60', '70', '80',
             '90', '91', '92', '93', '94', '95', '96', '97', '98',
             '99', 'Rows', 'Features'])

EVALUATED_FILE_REMOVED_ROWS_MEMORY_INFORMATION = pd.DataFrame(
    columns=['0', '10', '20', '30', '40', '50', '60', '70', '80',
             '90', '91', '92', '93', '94', '95', '96', '97', '98',
             '99', 'Rows', 'Features'])

# Predictions vs Real
EVALUATED_FILE_PREDICTED_VALUES = pd.DataFrame(columns=['y', 'y_hat'])

# General information about runtime predictions using a non modified data set
EVALUATED_FILE_RUNTIME_INFORMATION = pd.DataFrame(
    columns=['File Name', 'Train Score', 'Test Score', 'Potential Over Fitting', 'Initial Row Count',
             'Initial Feature Count', 'Processed Row Count', 'Processed Feature Count'])

# General information about memory predictions using a non modified data set
EVALUATED_FILE_MEMORY_INFORMATION = pd.DataFrame(
    columns=['File Name', 'Train Score', 'Test Score', 'Potential Over Fitting', 'Initial Row Count',
             'Initial Feature Count', 'Processed Row Count', 'Processed Feature Count'])
