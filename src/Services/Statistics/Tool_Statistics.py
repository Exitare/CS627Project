from RuntimeContants import Runtime_Datasets
import pandas as pd
import os
from RuntimeContants import Runtime_Folders
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from Services.Configuration.Config import Config
from RuntimeContants import Runtime_Datasets
import numpy as np
import math
from Services.Helper import Data_Frame_Helper

sns.set(style="whitegrid")


def create_additional_data():
    """
    Creates additional data not tied to a specific tool or file
    """
    __create_best_performing_version_per_tool_data_set()
    __create_worst_performing_version_per_tool_data_set()


def generate_tool_statistics():
    """
    Plots and prints additional data not tied to a specific tool or file
    """
    __count_best_split()
    __calculate_data_set_statistics()
    __get_best_performing_version_per_tool()
    __get_worst_performing_version_per_tool()
    __prediction_score_on_average_across_versions()
    __plot_predictions_result()


def __calculate_data_set_statistics():
    """
    Calculates the stats for all different data frames
    """
    whole_df = __calculate_whole_data_set_statistics()
    simple_df = __calculate_simple_data_set_statistics()

    frames = [whole_df, simple_df]
    df = pd.concat(frames)

    df.to_csv(Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, f"test_score_statistics.csv"),
              index=False)

    # Plot per label
    for label in Config.LABELS:
        data = df.loc[df['Label'] == label]

        if data.empty:
            continue

        # Reset index and delete the new created column
        data = data.reset_index()
        del data["index"]

        melt_df = pd.melt(data, id_vars=['Data'], value_vars=['Mean', 'Median', 'Correlation'])
        melt_df["Label"] = label

        for variable in melt_df['variable'].unique():
            temp_df = melt_df.loc[melt_df['variable'] == variable]
            ax = sns.barplot(x="variable", y="value", hue="Data", data=temp_df)
            # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set(xlabel=variable, ylabel='Value')
            fig = ax.get_figure()

            fig.savefig(
                Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, f"{label}_{variable}_test_score_statistics.jpg"),
                bbox_inches="tight")
            fig.clf()
            plt.close('all')


def __calculate_simple_data_set_statistics():
    """
    Calculates the statistics for the simple data sets
    """
    df = pd.DataFrame(columns=["Data", "Label", "Mean", "Median", "Correlation"])

    # Gather data
    temp_df = pd.DataFrame(
        columns=["Label", "File Name", "Train Score", "Test Score", "Potential Over Fitting", "Initial Row Count",
                 "Initial Feature Count", "Processed Row Count", "Processed Feature Count", "Features"])
    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        for file in tool.verified_files:
            for label, data in file.simple_dfs_evaluation.items():
                temp_df = temp_df.append(data)
                temp_df["Label"].fillna(label, inplace=True)
                temp_df["Test Score"].fillna(0, inplace=True)

    temp_df = temp_df.reset_index()
    # remove newly created index column
    del temp_df['index']

    # Split into labels and print them
    for label in Config.LABELS:
        data = temp_df.loc[temp_df['Label'] == label]

        if data.empty:
            continue

        df = df.append({"Data": "Simple Dataset", "Label": label, "Mean": data["Test Score"].mean(),
                        "Median": data["Test Score"].median(),
                        "Correlation": data["Test Score"].astype(float).corr(
                            data["Processed Feature Count"].astype(float))}, ignore_index=True)

    return df


def __calculate_whole_data_set_statistics():
    """
    Calculates the median and mean
    """

    df = pd.DataFrame(columns=["Data", "Label", "Mean", "Median", "Correlation"])

    for label in Runtime_Datasets.BEST_PERFORMING_VERSIONS:
        data = Runtime_Datasets.BEST_PERFORMING_VERSIONS[label]

        if data.empty:
            continue

        df = df.append({"Data": "Best Performing Version", "Label": label, "Mean": data["Test Score"].mean(),
                        "Median": data["Test Score"].median(),
                        "Correlation": data["Test Score"].corr(data["Processed Feature Count"])}, ignore_index=True)

    for label in Runtime_Datasets.WORST_PERFORMING_VERSIONS:
        data = Runtime_Datasets.WORST_PERFORMING_VERSIONS[label]

        if data.empty:
            continue

        df = df.append({"Data": "Worst Performing Version", "Label": label, "Mean": data["Test Score"].mean(),
                        "Median": data["Test Score"].median(),
                        "Correlation": data["Test Score"].corr(data["Processed Feature Count"])}, ignore_index=True)

    return df


def __create_best_performing_version_per_tool_data_set():
    """
    Creates a dataset containing the best performing versions of each evaluated tool
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

    Runtime_Datasets.BEST_PERFORMING_VERSIONS = performances


def __create_worst_performing_version_per_tool_data_set():
    """
    Creates a dataset containing the worst performing versions of each evaluated tool
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

    Runtime_Datasets.WORST_PERFORMING_VERSIONS = performances


def __get_best_performing_version_per_tool():
    """
    Get best performing version of each tool, sort them and write them as csv file
    """

    for label in Runtime_Datasets.BEST_PERFORMING_VERSIONS:
        if Runtime_Datasets.BEST_PERFORMING_VERSIONS[label].empty:
            continue

        data = Runtime_Datasets.BEST_PERFORMING_VERSIONS[label]
        data = __row_helper(data)

        data.sort_values(by=['Test Score'], inplace=True, ascending=False)
        data = data[
            ['Tool', 'File Name', 'Initial Feature Count', 'Initial Row Count', 'Potential Over Fitting',
             'Processed Feature Count', 'Processed Row Count', 'Test Score', 'Train Score']]
        data.to_csv(
            os.path.join(Runtime_Folders.EVALUATION_DIRECTORY,
                         f"tools_{label}_best_performing_by_version_by_test_score.csv"),
            index=False)

        data.sort_values(by='File Name', inplace=True)
        data.to_csv(
            os.path.join(Runtime_Folders.EVALUATION_DIRECTORY,
                         f"tools_{label}_best_performing_by_version_by_name.csv"),
            index=False)


def __get_worst_performing_version_per_tool():
    """
    Get worst performing version of each tool, sort them and write them as csv file
    """

    for label in Runtime_Datasets.WORST_PERFORMING_VERSIONS:
        if Runtime_Datasets.WORST_PERFORMING_VERSIONS[label].empty:
            continue

        data = Runtime_Datasets.WORST_PERFORMING_VERSIONS[label]
        data = __row_helper(data)

        data.sort_values(by=['Test Score'], inplace=True, ascending=False)
        data = data[
            ['Tool', 'File Name', 'Initial Feature Count', 'Initial Row Count', 'Potential Over Fitting',
             'Processed Feature Count', 'Processed Row Count', 'Test Score', 'Train Score']]
        data.to_csv(
            os.path.join(Runtime_Folders.EVALUATION_DIRECTORY,
                         f"tools_{label}_worst_performing_by_version_by_test_score.csv"),
            index=False)

        data.sort_values(by='File Name', inplace=True)
        data.to_csv(
            os.path.join(Runtime_Folders.EVALUATION_DIRECTORY,
                         f"tools_{label}_worst_performing_by_version_by_name.csv"),
            index=False)


def __prediction_score_on_average_across_versions():
    """
    Calculates the average predication rate (Test Score) across versions and labels
    NOT including the merged files
    """
    try:
        tool_avg_scores = dict()
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

                if label not in tool_avg_scores:
                    tool_avg_scores[label] = pd.DataFrame(
                        columns=["Tool", "Test Score", "Versions", "Average Rows"])
                    tool_avg_scores[label] = tool_avg_scores[label].append(
                        {"Tool": tool.name, "Test Score": test_scores[label].mean(),
                         "Versions": int(file_count),
                         "Average Rows": int(rows / file_count)}, ignore_index=True)
                else:
                    tool_avg_scores[label] = tool_avg_scores[label].append(
                        {"Tool": tool.name, "Test Score": test_scores[label].mean(),
                         "Versions": int(file_count),
                         "Average Rows": int(rows / file_count)}, ignore_index=True)

        for label in tool_avg_scores:
            if tool_avg_scores[label].empty:
                continue

            tool_avg_scores[label].sort_values(by="Test Score", ascending=False, inplace=True)
            tool_avg_scores[label].to_csv(
                Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, f"tools_{label}_test_score_on_average.csv"),
                index=False)

            print()

            # Plotting
            ax = sns.scatterplot(data=tool_avg_scores[label], x="Average Rows", y="Test Score", hue="Tool")
            ax.legend(bbox_to_anchor=(1.02, 1), loc=2)
            fig = ax.get_figure()

            fig.savefig(Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY,
                                      f"tools_{label}_test_score_on_average_score_by_rows.jpg"),
                        bbox_inches="tight")
            fig.clf()
            plt.close('all')

            ax = sns.scatterplot(data=tool_avg_scores[label], x="Versions", y="Test Score", hue="Tool")
            ax.legend(bbox_to_anchor=(1.02, 1), loc=2)
            fig = ax.get_figure()

            fig.savefig(Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY,
                                      f"tools_{label}_test_score_on_average_score_by_versions.jpg"),
                        bbox_inches="tight")
            fig.clf()
            plt.close('all')


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

        # Create a new column Version which only contains the version number
        data['Version'] = data.apply(
            lambda x: x["File Name"].split('_')[-1] if ("merged" not in x["File Name"]) else x["File Name"], axis=1)

        # Clean outliers
        data = data[np.abs(data["Test Score"] - data["Test Score"].mean()) <= (3 * data["Test Score"].std())]

        # how many plots should be created

        unique_tools = data['Tool'].unique()

        if len(unique_tools) > 4:
            part_df = pd.DataFrame(columns=['Tool'])
            i = 0
            for tool in unique_tools:
                if len(part_df['Tool']) >= 4:
                    fig = sns.FacetGrid(part_df, col="Tool", hue="Version", legend_out=True, col_wrap=2)
                    fig.map_dataframe(sns.scatterplot, x="Processed Feature Count", y="Test Score").add_legend()
                    fig.set_axis_labels("Processed Feature Count", "Test Score")
                    fig.savefig(
                        Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, f"{label}_prediction_overview_{i}.jpg"),
                        bbox_inches="tight")
                    part_df = pd.DataFrame()
                    i += 1

                part_df = part_df.append(data.loc[data['Tool'] == tool])

        else:
            fig = sns.FacetGrid(data, col="Tool", hue="Version", legend_out=True, col_wrap=3)
            fig.map_dataframe(sns.scatterplot, x="Processed Feature Count", y="Test Score").add_legend()
            fig.set_axis_labels("Processed Feature Count", "Test Score")
            fig.savefig(Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, f"{label}_prediction_overview.jpg"),
                        bbox_inches="tight")

        plt.close('all')


def __count_best_split():
    """
    Counts which split performs best and create a plot and dataset
    """

    versions = dict()
    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        for file in tool.verified_files:
            for label, data in file.split_evaluation_results.items():
                # Add index to keys
                for index in list(data.index):
                    if index not in versions:
                        versions[index] = 0

                index = file.get_best_performing_split(label)
                versions[index] += 1

    versions = {
        "version": list(versions.keys()),
        "count": list(versions.values())
    }

    versions = pd.DataFrame.from_dict(versions)

    versions.to_csv(Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, f"split_performance_count.csv"),
                    index=False)

    temp = versions.loc[versions['count'] != 0]

    ax = sns.barplot(x="version", y="count", data=temp)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    fig = ax.get_figure()

    fig.savefig(Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, f"split_performance_count.jpg"),
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
