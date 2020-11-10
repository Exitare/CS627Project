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

    # g = sns.catplot(x="time", y="pulse", hue="kind", col="diet", data=df, height=5, aspect=.8)

    # Plot per label
    for label in Config.LABELS:
        data = df.loc[df['Label'] == label]

        if data.empty:
            continue

        data = data.reset_index()
        del data["index"]

        print(data)
        melt_df = pd.melt(data, id_vars=['Data'], value_vars=['Mean', 'Median', 'Correlation'])
        print(melt_df)
        print(data["Data"])
        input()

        melt_df["Label"] = label
        print(melt_df)

        ax = sns.barplot(x="variable", y="value",  hue="Data", data=melt_df)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        fig = ax.get_figure()

        fig.savefig(Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, f"{label}_test_score_statistics.jpg"),
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
                temp_df.fillna(label, inplace=True)

    temp_df = temp_df.reset_index()
    # remove newly created index column
    del temp_df['index']

    # Split into labels and print them
    for label in Config.LABELS:
        data = temp_df.loc[temp_df['Label'] == label]

        if data.empty:
            continue

        df = df.append({"Data": "Simple Dataset", "Label": label, "Mean": data['Test Score'].mean(),
                        "Median": data['Test Score'].median(),
                        "Correlation": data["Test Score"].corr(data["Processed Feature Count"])}, ignore_index=True)

    return df


def __calculate_whole_data_set_statistics():
    """
    Calculates the median and mean
    """

    df = pd.DataFrame(columns=["Data", "Label", "Mean", "Median", "Correlation"])

    for label in Runtime_Datasets.BEST_PERFORMING_VERSIONS:
        data = Runtime_Datasets.BEST_PERFORMING_VERSIONS[label]
        df = df.append({"Data": "Best Performing Version", "Label": label, "Mean": data["Test Score"].mean(),
                        "Median": data["Test Score"].median(),
                        "Correlation": data['Test Score'].corr(data["Processed Feature Count"])}, ignore_index=True)

    for label in Runtime_Datasets.WORST_PERFORMING_VERSIONS:
        data = Runtime_Datasets.WORST_PERFORMING_VERSIONS[label]
        df = df.append({"Data": "Worst Performing Version", "Label": label, "Mean": data["Test Score"].mean(),
                        "Median": data["Test Score"].median(),
                        "Correlation": data['Test Score'].corr(data["Processed Feature Count"])}, ignore_index=True)

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
            os.path.join(Runtime_Folders.EVALUATION_DIRECTORY, f"tools_{label}_best_performing_by_version_by_name.csv"),
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


def __count_best_split():
    """
    Counts which split performs best and create a plot and dataset
    """

    versions = dict()
    version_count = pd.DataFrame()
    for tool in Runtime_Datasets.VERIFIED_TOOLS:
        for file in tool.verified_files:
            for label, data in file.split_evaluation_results.items():
                # Add index to keys
                for index in list(data.index):
                    if index not in version_count:
                        version_count[index] = 0
                    if index not in versions:
                        versions[index] = 0

                index = file.get_best_performing_split(label)
                versions[index] += 1

    versions = pd.Series(versions)

    version_count = version_count.append(versions, ignore_index=True)
    version_count.to_csv(Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, f"split_performance_count.csv"),
                         index=False)

    ax = sns.boxplot(x=0, data=version_count.T,
                     palette="Set3")
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
