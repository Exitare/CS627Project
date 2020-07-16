from RuntimeContants import Runtime_Datasets
import pandas as pd
import os
from RuntimeContants import Runtime_Folders


def get_best_performing_tools():
    """
    Get best performing version of each tool, sort them and write them as csv file
    """
    performance = pd.DataFrame()

    # Find best performing tool
    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        row = tool.get_best_performing_tool(True)

        if row is None:
            continue

        row['Tool'] = tool.name
        performance = performance.append(row)

    if performance.empty:
        return

    performance = __row_helper(performance)

    performance.sort_values(by=['Test Score'], inplace=True, ascending=False)
    performance = performance[
        ['Tool', 'File Name', 'Initial Feature Count', 'Initial Row Count', 'Potential Over Fitting',
         'Processed Feature Count', 'Processed Row Count', 'Test Score', 'Train Score']]
    performance.to_csv(os.path.join(Runtime_Folders.EVALUATION_DIRECTORY, "runtime_best_performing_tools.csv"),
                       index=False)

    performance = pd.DataFrame()

    # Find best performing tool
    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        row = tool.get_best_performing_tool(False)

        if row is None:
            continue

        row['Tool'] = tool.name
        performance = performance.append(row)

    if performance.empty:
        return

    performance = __row_helper(performance)

    performance.sort_values(by=['Test Score'], inplace=True, ascending=False)
    performance = performance[
        ['Tool', 'File Name', 'Initial Feature Count', 'Initial Row Count', 'Potential Over Fitting',
         'Processed Feature Count', 'Processed Row Count', 'Test Score', 'Train Score']]
    performance.to_csv(os.path.join(Runtime_Folders.EVALUATION_DIRECTORY, "memory_best_performing_tools.csv"),
                       index=False)


def get_worst_performing_tools():
    """
    Get worst performing version of each tool, sort them and write them as csv file
    """
    # Reset data frame
    performance = pd.DataFrame()

    # Find worst performing tools
    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        row = tool.get_worst_performing_tool(True)

        if row is None:
            continue

        row['Tool'] = tool.name
        performance = performance.append(row)

    if performance.empty:
        return

    performance = __row_helper(performance)

    performance.sort_values(by=['Test Score'], inplace=True, ascending=False)
    performance = performance[
        ['Tool', 'File Name', 'Initial Feature Count', 'Initial Row Count', 'Potential Over Fitting',
         'Processed Feature Count', 'Processed Row Count', 'Test Score', 'Train Score']]
    performance.to_csv(os.path.join(Runtime_Folders.EVALUATION_DIRECTORY, "runtime_worst_performing_tools.csv"),
                       index=False)

    performance = pd.DataFrame()

    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        row = tool.get_worst_performing_tool(False)

        if row is None:
            continue

        row['Tool'] = tool.name
        performance = performance.append(row)

    if performance.empty:
        return

    performance = __row_helper(performance)

    performance.sort_values(by=['Test Score'], inplace=True, ascending=False)
    performance = performance[
        ['Tool', 'File Name', 'Initial Feature Count', 'Initial Row Count', 'Potential Over Fitting',
         'Processed Feature Count', 'Processed Row Count', 'Test Score', 'Train Score']]
    performance.to_csv(os.path.join(Runtime_Folders.EVALUATION_DIRECTORY, "memory_worst_performing_tools.csv"),
                       index=False)


def __row_helper(performance_df):
    """
    Removes row, that shouldnt be in the data set.
    """
    if 'index' in performance_df:
        del performance_df['index']

    if 'level_0' in performance_df:
        del performance_df['level_0']

    return performance_df
