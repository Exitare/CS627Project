import matplotlib.pyplot as plt
import Constants
from Services import Config, NumpyHelper
import numpy as np


def tool_evaluation(df, action):
    plt.figure()
    ax = df.boxplot()
    labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '91', '92', '93', '94', '95', '96', '97',
              '98' '99']
    ax.set_ylabel('R2 Score')
    ax.set_xlabel("Percentage of rows removed")
    ax.set_title('R2 Values for 5K Fold')
    ax.set_xticklabels(labels)
    # ax.set_ylim([0, 1])
    # ax.set_yscale('log')
    plt.yscale('symlog')
    plt.savefig(f"{Constants.CURRENT_EVALUATED_TOOL_DIRECTORY}/{action}.jpg", dpi=None, format='png')
    plt.close()


def plot_summary():
    if not NumpyHelper.df_only_nan(Constants.RUNTIME_MEAN_REPORT):
        plot(Constants.RUNTIME_MEAN_REPORT, Constants.CURRENT_WORKING_DIRECTORY, "Average_Runtime_Mean_Report")
    if not NumpyHelper.df_only_nan(Constants.RUNTIME_VAR_REPORT):
        plot(Constants.RUNTIME_VAR_REPORT, Constants.CURRENT_WORKING_DIRECTORY, "Average_Runtime_Var_Report")
    if not NumpyHelper.df_only_nan(Constants.MEMORY_MEAN_REPORT):
        plot(Constants.MEMORY_MEAN_REPORT, Constants.CURRENT_WORKING_DIRECTORY, "Average_Memory_Mean_Report")
    if not NumpyHelper.df_only_nan(Constants.MEMORY_VAR_REPORT):
        plot(Constants.MEMORY_VAR_REPORT, Constants.CURRENT_WORKING_DIRECTORY, "Average_Memory_Var_Report")


def plot(df, path, file_name):
    plt.figure()
    ax = df.boxplot()
    labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '91', '92', '93', '94', '95', '96', '97',
              '98' '99']
    ax.set_ylabel('R2 Score')
    ax.set_xlabel("Percentage of rows removed")
    ax.set_title('R2 Values for 5K Fold')
    ax.set_xticklabels(labels)
    plt.yscale('symlog')
    plt.savefig(f"{path}/{file_name}.jpg", dpi=None, format='png')
    plt.close()


def plot_group_by(df, path, file_name, group_by):
    grouped = df.groupby(f'{group_by}')
    fig, ax = plt.subplots(figsize=(15, 7))
    # use unstack()
    df.groupby(['parameter_count']).plot(ax=ax)
    ax.get_legend().remove()
    plt.yscale('symlog')
    plt.show()
    # plt.yscale('symlog')
    # plt.savefig(f"{path}/{file_name}.jpg", dpi=None, format='png')
    # plt.close()


def plot_group_by_parameter_count():
    if not NumpyHelper.df_only_nan(Constants.RUNTIME_MEAN_REPORT):
        plot_group_by(Constants.RUNTIME_MEAN_REPORT, Constants.CURRENT_WORKING_DIRECTORY,
                      "Average_Runtime_Mean_Report_Grouped_By_Parameter_Count",
                      "parameter_count")
    if not NumpyHelper.df_only_nan(Constants.RUNTIME_VAR_REPORT):
        plot_group_by(Constants.RUNTIME_VAR_REPORT, Constants.CURRENT_WORKING_DIRECTORY,
                      "Average_Runtime_Var_Report_Grouped_By_Parameter_Count",
                      "parameter_count")
    if not NumpyHelper.df_only_nan(Constants.MEMORY_MEAN_REPORT):
        plot_group_by(Constants.MEMORY_MEAN_REPORT, Constants.CURRENT_WORKING_DIRECTORY,
                      "Average_Memory_Mean_Report_Grouped_By_Parameter_Count",
                      "parameter_count")
    if not NumpyHelper.df_only_nan(Constants.MEMORY_VAR_REPORT):
        plot_group_by(Constants.MEMORY_VAR_REPORT, Constants.CURRENT_WORKING_DIRECTORY,
                      "Average_Memory_Var_Report_Grouped_By_Parameter_Count",
                      "parameter_count")
