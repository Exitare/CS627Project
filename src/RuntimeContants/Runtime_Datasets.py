import pandas as pd

RUNTIME_MEAN_REPORT = []
RUNTIME_VAR_REPORT = []
MEMORY_MEAN_REPORT = []
MEMORY_VAR_REPORT = []
# Predictions vs Real
EVALUATED_FILE_PREDICTED_VALUES = []

OVER_UNDER_FITTING = pd.DataFrame(
    columns=['File Name', 'Train Score', 'Test Score', 'Potential Over Fitting', 'Row Count', 'Parameter Count'])


EVALUATED_FILE_NAMES = []
# Backwards compatibility
EVALUATED_FILE_ROW_COUNTS = []
EVALUATED_FILE_PARAMETER_COUNTS = []

# All Command line arguments passed at the start
COMMAND_LINE_ARGS = []