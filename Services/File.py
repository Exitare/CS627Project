import pandas as pd
import datetime
import os


def create_file(df, folder, name):
    now = datetime.datetime.now()
    if folder != "":
        path = os.path.join(folder, f"{name}.csv")
        df.to_csv(path, index=True)


def createFolder(filename):
    now = datetime.datetime.now()
    path = f"Results/{filename}{now.strftime('%Y-%m-%d-%H-%M-%S')}"
    try:
        os.mkdir(path)

    except OSError:
        print("Creation of the directory %s failed" % path)
        return ""
    else:
        print("Successfully created the directory %s " % path)
        return path
