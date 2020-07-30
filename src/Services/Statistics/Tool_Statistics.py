from RuntimeContants import Runtime_Datasets
import pandas as pd
import os
from RuntimeContants import Runtime_Folders
from pathlib import Path


def get_best_performing_tools():
    """
    Get best performing version of each tool, sort them and write them as csv file
    """
    performance = pd.DataFrame()

    # Find best performing tool
    # Runtime Performance
    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        row = tool.get_best_performing_version(True)

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
    performance.to_csv(
        os.path.join(Runtime_Folders.EVALUATION_DIRECTORY, "tools_runtime_best_performing_by_version.csv"),
        index=False)

    performance = pd.DataFrame()

    # Find best performing tool
    # Memory performance
    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        row = tool.get_best_performing_version(False)

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
    performance.to_csv(
        os.path.join(Runtime_Folders.EVALUATION_DIRECTORY, "tools_memory_best_performing_by_version.csv"),
        index=False)


def get_worst_performing_tools():
    """
    Get worst performing version of each tool, sort them and write them as csv file
    """
    # Runtime Performance
    performance = pd.DataFrame()

    # Find worst performing tools
    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        row = tool.get_worst_performing_version(True)

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
    performance.to_csv(
        os.path.join(Runtime_Folders.EVALUATION_DIRECTORY, "tools_runtime_worst_performing_by_version.csv"),
        index=False)

    performance = pd.DataFrame()

    # Memory performance
    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        row = tool.get_worst_performing_version(False)

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
    performance.to_csv(
        os.path.join(Runtime_Folders.EVALUATION_DIRECTORY, "tools_memory_worst_performing_by_version.csv"),
        index=False)


def prediction_score_on_average_across_versions(runtime: bool):
    """
    Calculates the average predication rate  (Test Score) across versions.
    NOT including the merged files
    """
    try:
        tool_scores = pd.DataFrame(columns=["Tool", "Test Score (avg)"])

        for tool in Runtime_Datasets.VERIFIED_TOOLS:
            test_scores = []
            for file in tool.verified_files:
                if not file.merged_file:
                    if runtime:
                        test_scores.append(file.runtime_evaluation['Test Score'])
                    else:
                        test_scores.append(file.memory_evaluation['Test Score'])

            tool_scores = tool_scores.append({"Tool": tool.name, "Test Score (avg)": pd.Series(test_scores).mean()},
                                             ignore_index=True)

        if runtime:
            tool_scores.to_csv(
                Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, "tools_runtime_test_score_on_average.csv"),
                index=False)
        else:
            tool_scores.to_csv(
                Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, "tools_memory_test_score_on_average.csv"),
                index=False)
    except:
        pass


def __row_helper(performance_df):
    """
    Removes rows, which should not be in the data set.
    """
    if 'index' in performance_df:
        del performance_df['index']

    if 'level_0' in performance_df:
        del performance_df['level_0']

    return performance_df
