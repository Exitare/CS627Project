from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from Services import PreProcessing, Config
from RuntimeContants import Runtime_File_Data, Runtime_Datasets, Runtime_Folders
from Services.Plotting import Plotting_Data_Removal
from Services.File import General_File_Service
import sys


def removal_helper():
    """
    Helper for managing all task related to the data removal, prediction etc
    :return:
    """
    try:
        df = Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET.copy()
        if 'runtime' in df.columns:
            print("Predicting runtime...")
            scores = predict(df, 'runtime')
            Plotting_Data_Removal.tool_evaluation(scores, "runtime")
            General_File_Service.create_csv_file(scores, Runtime_Folders.CURRENT_EVALUATED_TOOL_DIRECTORY,
                                                 "data_removal_runtime_evaluation")

        df = Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET.copy()
        if 'memory.max_usage_in_bytes' in df.columns:
            print("Predicting memory...")
            scores = predict(df, 'memory.max_usage_in_bytes')
            Plotting_Data_Removal.tool_evaluation(scores, "memory")
            General_File_Service.create_csv_file(scores, Runtime_Folders.CURRENT_EVALUATED_TOOL_DIRECTORY,
                                                 "data_removal_memory_evaluation")


    except BaseException as ex:
        print(ex)
        General_File_Service.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
        sys.exit()


def predict(df, feature: str):
    averages_per_repetition = pd.DataFrame(
        columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '91', '92', '93', '94',
                 '95', '96', '97', '98', '99'])

    final_scores = pd.DataFrame(
        columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '91', '92', '93', '94',
                 '95', '96', '97', '98', '99'])

    y = df[f'{feature}']
    del df[f'{feature}']
    X = df
    X = PreProcessing.normalize_X(X)

    for i in range(0, Config.Config.REPETITIONS, 1):
        print(f"Started repetition # {i + 1}")
        averages_per_repetition = averages_per_repetition.append(k_folds(X, y))
        final_scores = final_scores.append(averages_per_repetition.mean(), ignore_index=True)

    return final_scores


def k_folds(X, y):
    """
    Trains the model to predict the total time
    :param X:
    :param y:
    :return:
    """
    try:
        model = RandomForestRegressor(n_estimators=Config.Config.FOREST_ESTIMATORS, random_state=1)

        kf = KFold(n_splits=Config.Config.K_FOLDS)
        scores = pd.DataFrame(columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '91', '92', '93',
                                       '94',
                                       '95', '96', '97', '98', '99'])
        counter = 0
        for train_index, test_index in kf.split(X):
            r2scores = []
            # Iterate from 0 to 101. Ends @ 100, reduce by 1
            for i in range(0, 101, 1):
                if i <= 90 and i % 10 == 0:
                    r2scores.append(calculate(model, i, X, y, train_index, test_index))
                if 99 >= i > 90:
                    r2scores.append(calculate(model, i, X, y, train_index, test_index))

            scores = scores.append(
                {'0': r2scores[0], '10': r2scores[1], '20': r2scores[2], '30': r2scores[3], '40': r2scores[4],
                 '50': r2scores[5], '60': r2scores[6], '70': r2scores[7], '80': r2scores[8],
                 '90': r2scores[9], '91': r2scores[10], '92': r2scores[11], '93': r2scores[12],
                 '94': r2scores[13], '95': r2scores[14], '96': r2scores[15], '97': r2scores[16],
                 '98': r2scores[17], '99': r2scores[18]}, ignore_index=True)

            counter += 1

        return scores

    except BaseException as ex:
        print(ex)
        scores = pd.DataFrame(0, index=np.arange(Config.Config.K_FOLDS),
                              columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '91', '92', '93',
                                       '94',
                                       '95', '96', '97', '98', '99'])
        return scores


def calculate(model, i, X, y, train_index, test_index):
    """
    Calculates the r2 scores
    :param model:
    :param i:
    :param X:
    :param y:
    :param train_index:
    :param test_index:
    :return:
    """
    # Create a deep copy, to keep the original data set untouched
    train_index_copy = train_index

    source_len = len(train_index_copy)

    # Calculate amount of rows to be removed
    rows = int(len(train_index_copy) * i / 100)

    # Remove rows by random index
    train_index_copy = PreProcessing.remove_random_rows(train_index_copy, rows)
    # print(f"Removed {rows} rows, {len(train_index_copy)} indices left of {source_len}. {100 - i}% data left!")

    X_train, X_test = X[train_index_copy], X[test_index]
    y_train, y_test = y[train_index_copy], y[test_index]

    # print(f"y_test contains {len(y_test)} rows")
    # print(f"X_test contains {len(X_test)} rows")

    # print(f"y_train contains {len(y_train)} rows")
    # print(f"X_train contains {len(X_train)} rows")

    model.fit(X_train, y_train)
    y_test_hat = model.predict(X_test)
    return r2_score(y_test, y_test_hat)