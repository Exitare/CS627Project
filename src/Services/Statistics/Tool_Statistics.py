from RuntimeContants import Runtime_Datasets
import pandas as pd
import os
from RuntimeContants import Runtime_Folders
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def generate_tool_statistics():
    __get_best_performing_tools()
    __get_worst_performing_tools()
    __prediction_score_on_average_across_versions(True)
    __prediction_score_on_average_across_versions(False)
    __plot_predictions_result()


def __get_best_performing_tools():
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


def __get_worst_performing_tools():
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


def __prediction_score_on_average_across_versions(runtime: bool):
    """
    Calculates the average predication rate  (Test Score) across versions.
    NOT including the merged files
    """
    try:
        tool_scores = pd.DataFrame(columns=["Tool", "Test Score (avg)"])

        for tool in Runtime_Datasets.VERIFIED_TOOLS:
            test_scores = []
            # Get all files which are verified but not "merged" files
            files = [file for file in tool.verified_files if not file.merged_file]
            # helper for calculating the average row count of all versions
            rows = 0
            # how many files are added to the test scores, in case an evaluation is empty.
            file_count = 0
            for file in files:
                if runtime:
                    if not file.runtime_evaluation.empty:
                        test_scores.append(file.runtime_evaluation['Test Score'])
                        rows += file.get_pre_processed_df_statistics()[1]
                        file_count += 1
                else:
                    if not file.memory_evaluation.empty:
                        test_scores.append(file.memory_evaluation['Test Score'])
                        rows += file.get_pre_processed_df_statistics()[1]
                        file_count += 1

            # if no test scores are added, skip generating of the file
            if len(test_scores) == 0:
                return

            tool_scores = tool_scores.append(
                {"Tool": tool.name, "Test Score (avg)": pd.Series(test_scores).mean(), "Versions": int(file_count),
                 "Average Rows": int(rows / file_count)}, ignore_index=True)

        # Check if any tool scores where gathered
        if len(tool_scores) == 0:
            return

        if runtime:
            tool_scores.sort_values(by="Test Score (avg)", ascending=False, inplace=True)
            tool_scores.to_csv(
                Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, "tools_runtime_test_score_on_average.csv"),
                index=False)
        else:
            tool_scores.sort_values(by="Test Score (avg)", ascending=False, inplace=True)
            tool_scores.to_csv(
                Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, "tools_memory_test_score_on_average.csv"),
                index=False)
    except BaseException as ex:
        print("Exception occurred in prediction_score_on_average_across_versions")
        logging.error(ex)
        pass


def __plot_predictions_result():
    """

    """
    # combined runtime data of all tools for plotting
    combined_runtime_df = pd.DataFrame()
    # combined memory data of all tools for plotting
    combined_memory_df = pd.DataFrame()
    temp_df = []
    for tool in Runtime_Datasets.VERIFIED_TOOLS:

        if not tool.files_runtime_overview.empty:
            # Deep copy for manipulation
            runtime_copy = tool.files_runtime_overview.copy()
            runtime_copy['Tool'] = tool.name
            temp_df.append(runtime_copy)

    if len(temp_df) > 0:
        combined_runtime_df = pd.concat(temp_df, join='inner')

    # Clear temp_df
    temp_df = []

    for tool in Runtime_Datasets.VERIFIED_TOOLS:

        if not tool.files_memory_overview.empty:
            # Deep copy for manipulation
            runtime_copy = tool.files_memory_overview.copy()
            runtime_copy['Tool'] = tool.name
            temp_df.append(runtime_copy)

    if len(temp_df) > 0:
        combined_memory_df = pd.concat(temp_df, join='inner')

    if not combined_runtime_df.empty:
        ax = sns.boxplot(x="Tool", y="Test Score", data=combined_runtime_df,
                         palette="Set3")
        ax = sns.swarmplot(x="Tool", y="Test Score", data=combined_runtime_df, color=".25")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        fig = ax.get_figure()

        fig.savefig(Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, "runtime_prediction_overview"),
                    bbox_inches="tight")
        fig.clf()
        plt.close('all')

    if not combined_memory_df.empty:
        ax = sns.boxplot(x="Tool", y="Test Score", data=combined_memory_df,
                         palette="Set3")
        ax = sns.swarmplot(x="Tool", y="Test Score", data=combined_memory_df, color=".25")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        fig = ax.get_figure()
        fig.savefig(Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, "memory_prediction_overview"),
                    bbox_inches="tight")
        fig.clf()
        plt.close('all')


def __row_helper(performance_df):
    """
    Removes rows, which should not be in the data set.
    """
    if 'index' in performance_df:
        del performance_df['index']

    if 'level_0' in performance_df:
        del performance_df['level_0']

    return performance_df
