from RuntimeContants import Runtime_Datasets
import pandas as pd
import os
from RuntimeContants import Runtime_Folders
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from Services.Configuration.Config import Config

sns.set(style="whitegrid")


def generate_tool_statistics():
    __get_best_performing_tools()
    __get_worst_performing_tools()
    __prediction_score_on_average_across_versions()
    __plot_predictions_result()


def __get_best_performing_tools():
    """
    Get best performing version of each tool, sort them and write them as csv file
    """

    performances = dict()

    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        for label in Config.LABELS:
            version = tool.get_best_performing_version(label)

            if version is None:
                continue

            # Add the tool name to the version row
            version['Tool'] = tool.name

            if label in performances:
                performances[label] = performances[label].append(version)
            else:
                performances[label] = pd.DataFrame()
                performances[label] = performances[label].append(version)

    for label in performances:
        if performances[label].empty:
            continue

        label_performance = performances[label]
        label_performance = __row_helper(label_performance)

        label_performance.sort_values(by=['Test Score'], inplace=True, ascending=False)
        label_performance = label_performance[
            ['Tool', 'File Name', 'Initial Feature Count', 'Initial Row Count', 'Potential Over Fitting',
             'Processed Feature Count', 'Processed Row Count', 'Test Score', 'Train Score']]
        label_performance.to_csv(
            os.path.join(Runtime_Folders.EVALUATION_DIRECTORY, f"tools_{label}_best_performing_by_version.csv"),
            index=False)


def __get_worst_performing_tools():
    """
    Get worst performing version of each tool, sort them and write them as csv file
    """
    performances = dict()

    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        for label in Config.LABELS:
            version = tool.get_worst_performing_version(label)

            if version is None:
                continue

            # Add the tool name to the version row
            version['Tool'] = tool.name

            if label in performances:
                performances[label] = performances[label].append(version)
            else:
                performances[label] = pd.DataFrame()
                performances[label] = performances[label].append(version)

    for label in performances:
        if performances[label].empty:
            continue

        label_performance = performances[label]
        label_performance = __row_helper(label_performance)

        label_performance.sort_values(by=['Test Score'], inplace=True, ascending=False)
        label_performance = label_performance[
            ['Tool', 'File Name', 'Initial Feature Count', 'Initial Row Count', 'Potential Over Fitting',
             'Processed Feature Count', 'Processed Row Count', 'Test Score', 'Train Score']]
        label_performance.to_csv(
            os.path.join(Runtime_Folders.EVALUATION_DIRECTORY, f"tools_{label}_worst_performing_by_version.csv"),
            index=False)


def __prediction_score_on_average_across_versions():
    """
    Calculates the average predication rate (Test Score) across versions and labels
    NOT including the merged files
    """
    try:
        tool_scores = dict()
        pd.DataFrame(columns=["Tool", "Test Score (avg)", "Versions", " Average Rows"])
        for tool in Runtime_Datasets.VERIFIED_TOOLS:

            test_scores = dict()
            # Get all files which are verified but not "merged" files
            files = [file for file in tool.verified_files if not file.merged_file]
            # helper for calculating the average row count of all versions
            rows = 0
            # how many files are added to the test scores, in case an evaluation is empty.
            file_count = 0

            for file in files:
                for label in file.detected_labels:
                    # Check if label is present in test_scores
                    if label not in test_scores:
                        test_scores[label] = pd.Series()

                    if label in file.evaluation_results:
                        test_scores[label] = test_scores[label].append(file.evaluation_results[label]['Test Score'])
                        rows += file.get_pre_processed_df_statistics()[1]
                        file_count += 1

            # Merge gathered data together
            for label in Config.LABELS:
                if label not in test_scores:
                    continue

                if label not in tool_scores:
                    tool_scores[label] = pd.DataFrame(columns=["Tool", "Test Score (avg)", "Versions", "Average Rows"])
                    tool_scores[label] = tool_scores[label].append(
                        {"Tool": tool.name, "Test Score (avg)": test_scores[label].mean(),
                         "Versions": int(file_count),
                         "Average Rows": int(rows / file_count)}, ignore_index=True)
                else:
                    tool_scores[label] = tool_scores[label].append(
                        {"Tool": tool.name, "Test Score (avg)": test_scores[label].mean(),
                         "Versions": int(file_count),
                         "Average Rows": int(rows / file_count)}, ignore_index=True)

        for label in tool_scores:
            if tool_scores[label].empty:
                continue

            tool_scores[label].sort_values(by="Test Score (avg)", ascending=False, inplace=True)
            tool_scores[label].to_csv(
                Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, f"tools_{label}_test_score_on_average.csv"),
                index=False)

    except BaseException as ex:
        logging.exception(ex)


def __plot_predictions_result():
    """
    Plots the predictions results as bar graph
    """

    predictions_per_label = dict()
    temp_data_sets = dict()
    for tool in Runtime_Datasets.VERIFIED_TOOLS:

        for label in Config.LABELS:
            if label not in tool.files_label_overview:
                continue

            data = tool.files_label_overview[label].copy()
            data['Tool'] = tool.name

            if label not in temp_data_sets:
                temp_data_sets[label] = list()
                temp_data_sets[label].append(data)
            else:
                temp_data_sets[label].append(data)

    for label in Config.LABELS:
        if label not in temp_data_sets or len(temp_data_sets[label]) == 0:
            continue

        if label not in predictions_per_label:
            predictions_per_label[label] = pd.DataFrame()

        predictions_per_label[label] = pd.concat(temp_data_sets[label], ignore_index=True)

    for label in Config.LABELS:
        if label not in predictions_per_label:
            continue
        data = predictions_per_label[label]
        ax = sns.boxplot(x="Tool", y="Test Score", data=data,
                         palette="Set3")
        ax = sns.swarmplot(x="Tool", y="Test Score", data=data, color=".25")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        fig = ax.get_figure()

        fig.savefig(Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, f"{label}_prediction_overview.jpg"),
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
