import pandas as pd
from Services.Configuration.Config import Config
from RuntimeContants import Runtime_Datasets, Runtime_Folders
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from Services.Helper import Data_Frame_Helper

sns.set(style="whitegrid")


class ToolStatistics:
    all_tools_test_score_evaluation = pd.DataFrame()
    all_tools_performance_difference = pd.DataFrame(columns=["Tool", "Difference", 'Label'])
    multi_tools_performance_difference = pd.DataFrame(columns=["Tool", "Difference", 'Label'])
    whole_data_set_test_scores = pd.DataFrame(columns=["Source", "Tool", "Label", "Mean", "Median", "Correlation"])
    simple_data_set_test_scores = pd.DataFrame(columns=["Source", "Tool", "Label", "Mean", "Median", "Correlation"])

    def __init__(self):
        """
        All tool statistic calculation are done here. No plotting!
        """
        self.__gather_all_tool_evaluations()
        self.__calculate_tool_performance_difference()
        self.__calculate_simple_data_set_statistics()
        self.__calculate_whole_data_set_statistics()

    def __gather_all_tool_evaluations(self):
        """
        Concat all evaluations into one big file
        """
        for tool in Runtime_Datasets.VERIFIED_TOOLS:
            for file in tool.verified_files:
                data = file.simple_dfs_evaluation.copy()
                data["Origin"] = "Simple Data Set"
                self.all_tools_test_score_evaluation = self.all_tools_test_score_evaluation.append(data)

                data = file.evaluation_results.copy()
                data["Origin"] = "Whole Data Set"
                self.all_tools_test_score_evaluation = self.all_tools_test_score_evaluation.append(data)

                data = file.split_evaluation_results.copy()
                data["Origin"] = "Split Data Set"
                self.all_tools_test_score_evaluation = self.all_tools_test_score_evaluation.append(data)

    def __calculate_tool_performance_difference(self):
        """
        Calculates the difference between the best and worst performing version of each tool
        """

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
                self.all_tools_performance_difference = self.all_tools_performance_difference.append(
                    {
                        "Tool": tool.name,
                        "Difference": difference,
                        "Label": label
                    },
                    ignore_index=True)

                if len(tool.verified_files) > 1:
                    self.multi_tools_performance_difference = self.multi_tools_performance_difference.append(
                        {
                            "Tool": tool.name,
                            "Difference": difference,
                            "Label": label
                        },
                        ignore_index=True)

    def __calculate_whole_data_set_statistics(self):
        """
        Calculates the median, mean and correlation of all whole data sets
        """

        # Gathers all test score statistics
        for tool in Runtime_Datasets.VERIFIED_TOOLS:
            for file in tool.verified_files:
                for label in file.evaluation_results["Label"].unique():
                    data = Data_Frame_Helper.get_label_data(file.evaluation_results, label)
                    if data.empty:
                        continue

                    self.whole_data_set_test_scores = self.whole_data_set_test_scores.append(
                        {
                            "Source": "Whole Dataset",
                            "Label": label,
                            "Tool": tool,
                            "Mean": data["Test Score"].mean(),
                            "Median": data["Test Score"].median(),
                            "Correlation": data["Test Score"].astype(float).corr(
                                data["Processed Feature Count"].astype(float))
                        }, ignore_index=True)

    def __calculate_simple_data_set_statistics(self):
        """
        Calculates the statistics for the simple data sets
        """

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
                self.simple_data_set_test_scores = self.simple_data_set_test_scores.append(
                    {
                        "Source": "Simple Dataset",
                        "Label": label,
                        "Tool": tool,
                        "Mean": tool_data["Test Score"].mean(),
                        "Median": tool_data["Test Score"].median(),
                        "Correlation": tool_data["Test Score"].astype(float).corr(
                            tool_data["Processed Feature Count"].astype(float))
                    }, ignore_index=True)

    def plot(self):
        """
        Calls all plot functions
        """
        self.__plot_tools_performance_difference()
        self.__plot_test_scores()

    def __plot_tools_performance_difference(self):
        """
        Plots the tools performance difference
        """
        if not self.all_tools_performance_difference.empty:
            ax = sns.violinplot(x="Label", y="Difference", data=self.all_tools_performance_difference)
            fig = ax.get_figure()

            fig.savefig(
                Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, f"test_score_difference_all_tools.jpg"),
                bbox_inches="tight")
            fig.clf()
            plt.close('all')

        if not self.multi_tools_performance_difference.empty:
            ax = sns.violinplot(x="Label", y="Difference", data=self.multi_tools_performance_difference)
            fig = ax.get_figure()

            fig.savefig(
                Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, f"test_score_difference_multi_tools.jpg"),
                bbox_inches="tight")
            fig.clf()
            plt.close('all')

    def __plot_test_scores(self):
        """
        Plot all r2 scores for all tools
        """

        for origin in self.all_tools_test_score_evaluation["Origin"]:
            data = self.all_tools_test_score_evaluation[self.all_tools_test_score_evaluation["Origin"] == origin].copy()

            ax = sns.violinplot(x="Label", y="Test Score", data=data)
            fig = ax.get_figure()

            fig.savefig(
                Path.joinpath(Runtime_Folders.EVALUATION_DIRECTORY, f"{origin}_r2_scores.jpg"),
                bbox_inches="tight")
            fig.clf()
            plt.close('all')
