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
from scipy import stats

sns.set(style="whitegrid")


def get_all_tool_evaluations():
    """
    Concat all evaluations into one big file
    """

    all_tools_evaluation = pd.DataFrame()

    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        for file in tool.verified_files:
            data = file.simple_dfs_evaluation.copy()
            data["Origin"] = "Simple Data Set"
            data["Tool"] = tool.name
            all_tools_evaluation = all_tools_evaluation.append(data)

            data = file.evaluation_results.copy()
            data["Origin"] = "Whole Data Set"
            data["Tool"] = tool.name
            all_tools_evaluation = all_tools_evaluation.append(data)

            data = file.split_evaluation_results.copy()
            data["Origin"] = "Split Data Set"
            data["Tool"] = tool.name
            all_tools_evaluation = all_tools_evaluation.append(data)

    return all_tools_evaluation


def calculate_tool_performance_difference():
    """
    Calculates the difference between the best and worst performing version of each tool
    """

    all_tools_performance_difference = pd.DataFrame()
    multi_tools_performance_difference = pd.DataFrame()

    for label in Config.LABELS:
        for tool in Runtime_Datasets.VERIFIED_TOOLS:

            if tool.get_best_performing_version(label) is None or tool.get_worst_performing_version(label) is None:
                continue

            best_performing_version = tool.get_best_performing_version(label)['Test Score']
            worst_performing_version = tool.get_worst_performing_version(label)[
                'Test Score']

            if best_performing_version is None or worst_performing_version is None:
                continue

            difference = best_performing_version - worst_performing_version
            all_tools_performance_difference = all_tools_performance_difference.append(
                {
                    "Tool": tool.name,
                    "Difference": difference,
                    "Label": label
                },
                ignore_index=True)

            if len(tool.verified_files) > 1:
                multi_tools_performance_difference = multi_tools_performance_difference.append(
                    {
                        "Tool": tool.name,
                        "Difference": difference,
                        "Label": label
                    },
                    ignore_index=True)

    return all_tools_performance_difference, multi_tools_performance_difference


def calculate_whole_data_set_statistics():
    """
    Calculates the median, mean and correlation of all whole data sets
    """

    whole_data_set_test_scores = pd.DataFrame()

    # Gathers all test score statistics
    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        for file in tool.verified_files:
            for label in file.evaluation_results["Label"].unique():
                data = Data_Frame_Helper.get_label_data(file.evaluation_results, label)
                if data.empty:
                    continue

                whole_data_set_test_scores = whole_data_set_test_scores.append(
                    {
                        "Source": "Whole Dataset",
                        "Label": label,
                        "Tool": tool,
                        "Mean": data["Test Score"].mean(),
                        "Median": data["Test Score"].median(),
                        "Correlation": data["Test Score"].astype(float).corr(
                            data["Processed Feature Count"].astype(float))
                    }, ignore_index=True)

    return whole_data_set_test_scores


def calculate_simple_data_set_statistics():
    """
    Calculates the statistics for the simple data sets
    """

    simple_data_set_test_scores = pd.DataFrame()

    # Gather data
    temp_df = pd.DataFrame(
        columns=["Label", "Tool", "File Name", "Train Score", "Test Score", "Potential Over Fitting",
                 "Initial Row Count", "Initial Feature Count", "Processed Row Count", "Processed Feature Count"])
    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        for file in tool.verified_files:
            for label in file.simple_dfs_evaluation["Label"].unique():
                data = Data_Frame_Helper.get_label_data(file.simple_dfs_evaluation, label)
                temp_df = temp_df.append(data)
                temp_df["Tool"].fillna(tool.name, inplace=True)

    temp_df = temp_df.reset_index()
    # remove newly created index column
    del temp_df['index']

    # Split into labels
    for label in temp_df["Label"].unique():
        data = temp_df.loc[temp_df['Label'] == label]

        if data.empty:
            continue

        for tool in data["Tool"].unique():
            tool_data = data[data["Tool"] == tool]
            simple_data_set_test_scores = simple_data_set_test_scores.append(
                {
                    "Source": "Simple Dataset",
                    "Label": label,
                    "Tool": tool_data["Tool"],
                    "Mean": tool_data["Test Score"].mean(),
                    "Median": tool_data["Test Score"].median(),
                    "Correlation": tool_data["Test Score"].astype(float).corr(
                        tool_data["Processed Feature Count"].astype(float))
                }, ignore_index=True)

    return simple_data_set_test_scores


def get_all_merged_files_evaluations():
    """
    Gathers all merged files
    """
    all_merged_files_evaluations = pd.DataFrame()

    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        for file in tool.verified_files:
            if not file.merged_file:
                continue

            all_merged_files_evaluations = all_merged_files_evaluations.append(file.evaluation_results)
            all_merged_files_evaluations = all_merged_files_evaluations.append(file.simple_dfs_evaluation)
            all_merged_files_evaluations = all_merged_files_evaluations.append(file.split_evaluation_results)

    return all_merged_files_evaluations


def get_tools_above_threshold():
    data = pd.DataFrame()

    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        versions = pd.DataFrame()

    return pd.DataFrame()
