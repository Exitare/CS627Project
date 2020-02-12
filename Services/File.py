import pandas as pd
import datetime


def create_file(df):
    now = datetime.datetime.now()
    df.to_csv(f"{now.strftime('%Y-%m-%d %H-%M-%S')}-scores.csv", index=True)
