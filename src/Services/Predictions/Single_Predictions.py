from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from Services import Config
from Services.File import General_File_Service
from Services.Plotting import Single_Plotting
import pandas as pd
from RuntimeContants import Runtime_Folders, Runtime_Datasets, Runtime_File_Data
import sys
from sklearn.feature_selection import f_regression
from sklearn.metrics import r2_score


def compare_real_to_predicted_data(df):
    """
    Creates a plot which contains the real data compared to the predicted values
    :param df:
    :return:
    """
    try:
        if 'runtime' in df.columns:
            predict_runtime(df)

        if 'memory.max_usage_in_bytes' in df.columns:
            predict_memory(df)

    except BaseException as ex:
        print("Error in compare_real_to_predicted_data")
        print(ex)
        General_File_Service.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
        sys.exit()


def predict_runtime(df):
    value_comparison = pd.DataFrame(columns=['y', 'y_test_hat'])

    model = RandomForestRegressor(n_estimators=Config.Config.FOREST_ESTIMATORS, random_state=1)
    copy = df
    y = copy['runtime']
    del copy['runtime']
    X = copy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)

    model.fit(X_train, y_train)
    y_test_hat = model.predict(X_test)
    y_train_hat = model.predict(X_train)

    test_score = r2_score(y_test, y_test_hat)
    train_score = r2_score(y_train, y_train_hat)

    overFitting = False
    if train_score > test_score * 2:
        overFitting = True

    Runtime_Datasets.OVER_UNDER_FITTING = Runtime_Datasets.OVER_UNDER_FITTING.append(
        {'File Name': Runtime_File_Data.EVALUATED_FILE_NAME, "Test Score": test_score,
         "Train Score": train_score, "Potential Over Fitting": overFitting,
         "Row Count": Runtime_File_Data.EVALUATED_FILE_ROW_COUNT,
         "Parameter Count": Runtime_File_Data.EVALUATED_FILE_COLUMN_COUNT}, ignore_index=True)

    value_comparison = value_comparison.assign(y=pd.Series(y_test))
    value_comparison = value_comparison.assign(y_test_hat=pd.Series(y_test_hat))
    # Na values are set to 0
    value_comparison = value_comparison.fillna(0)

    f_regression(X, y)
    Single_Plotting.plot(value_comparison, "y_vs_y_hat")


def predict_memory(df):
    value_comparison = pd.DataFrame(columns=['y', 'y_test_hat'])

    model = RandomForestRegressor(n_estimators=Config.Config.FOREST_ESTIMATORS, random_state=1)
    copy = df
    y = copy['memory.max_usage_in_bytes']
    del copy['memory.max_usage_in_bytes']
    X = copy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
    model.fit(X_train, y_train)
    y_test_hat = model.predict(X_test)

    value_comparison.iloc['y'] = X_test
    value_comparison.iloc['y_test_hat'] = y_test_hat
