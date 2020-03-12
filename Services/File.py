import pandas as pd
import datetime
import os
import ntpath
import sys


def create_file(df, folder, name):
    now = datetime.datetime.now()
    if folder != "":
        path = os.path.join(folder, f"{name}.csv")
        df.to_csv(path, index=True)


def create_folder(args):

    check_results_folder_exists()

    now = datetime.datetime.now()
    path = f"Results/{now.strftime('%Y-%m-%d-%H-%M-%S')}-{get_file_name(args.filename)}"
    try:
        os.mkdir(path)

    except OSError:
        print("Creation of the directory %s failed" % path)
        print("Stopping application")
        sys.exit()
    else:
        print("Successfully created the directory %s " % path)
        return path


def get_file_name(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def check_results_folder_exists():
    if not os.path.isdir("Results/"):
        print(f"Creating results folder")
        os.mkdir("Results/")
