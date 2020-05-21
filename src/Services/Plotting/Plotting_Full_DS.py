import matplotlib.pyplot as plt
import numpy as np
from RuntimeContants import Runtime_Folders
from Services.FileSystem import General_File_Service
import sys


def plot(df, file_name):
    """Plots the df
    """
    try:
        df.plot.scatter(x='y', y='y_test_hat')

        fig, ax = plt.subplots()

        ax.set_title('Y vs Y^')
        ax.scatter(df['y'], df['y_test_hat'], label='Y',
                   alpha=0.5, edgecolors='none')
        ax.legend()
        ax.set_xlabel('Y')
        ax.set_ylabel('Y Hat')
        plt.yscale('symlog')
        plt.xscale('symlog')
        plt.savefig(f"{Runtime_Folders.CURRENT_EVALUATED_TOOL_DIRECTORY}/{file_name}.jpg", dpi=None, format='png')
        plt.close()
        plt.close('all')
    except BaseException as ex:
        print(ex)
        General_File_Service.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
        sys.exit()
