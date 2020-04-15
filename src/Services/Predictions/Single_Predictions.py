from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from Services import Config
from Services.File import General_File_Service
from Services.Plotting import Plotting_Full_DS
import pandas as pd
from RuntimeContants import Runtime_Folders, Runtime_Datasets, Runtime_File_Data
import sys
from sklearn.feature_selection import f_regression
from sklearn.metrics import r2_score
from Services import PreProcessing


def compare_real_to_predicted_data():
    """
    Creates a plot which contains the real data compared to the predicted values

    :return:
    """
    try:

        df = Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET
        if 'runtime' in df.columns:
            predict(df, 'runtime')

        if 'memory.max_usage_in_bytes' in df.columns:
            predict(df, 'max_usage_in_bytes')

    except BaseException as ex:
        print("Error in compare_real_to_predicted_data")
        print(ex)
        General_File_Service.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
        sys.exit()


def predict(df, feature: str):
    value_comparison = pd.DataFrame(columns=['y', 'y_test_hat'])

    model = RandomForestRegressor(n_estimators=Config.Config.FOREST_ESTIMATORS, random_state=1)
    # create a copy of the data set, because the df should be reused in other
    y = df[f'{feature}']
    del df[f'{feature}']
    X = df

    temp_len = len(X)

    X_indexes = (X != 0).any(axis=1)
    X = X.loc[X_indexes]
    y = y.loc[X_indexes]

    if temp_len != len(X):
        print(df)
        print(f"Removed {temp_len - len(X)} rows. Source had {temp_len}")

    # print(X)
    if len(X.index) == 0:
        General_File_Service.remove_folder(Runtime_Folders.CURRENT_EVALUATED_TOOL_DIRECTORY)
        return

    X = PreProcessing.normalize_X(X)
    X = PreProcessing.variance_selection(X)

    # Check if x is valid or not
    if type(X) == int:
        if X == 0:
            General_File_Service.remove_folder(Runtime_Folders.CURRENT_EVALUATED_TOOL_DIRECTORY)
            return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)

    model.fit(X_train, y_train)
    y_test_hat = model.predict(X_test)
    y_train_hat = model.predict(X_train)

    test_score = r2_score(y_test, y_test_hat)
    train_score = r2_score(y_train, y_train_hat)

    overFitting = False
    if train_score > test_score * 2:
        overFitting = True

    # TODO: Hardcoded hack, just be dynamic
    if feature == 'runtime':
        Runtime_Datasets.GENERAL_INFORMATION_RUNTIME = Runtime_Datasets.GENERAL_INFORMATION_RUNTIME.append(
            {'File Name': Runtime_File_Data.EVALUATED_FILE_NAME, "Test Score": test_score,
             "Train Score": train_score, "Potential Over Fitting": overFitting,
             "Initial Row Count": Runtime_File_Data.EVALUATED_FILE_ROW_COUNT,
             "Initial Feature Count": Runtime_File_Data.EVALUATED_FILE_COLUMN_COUNT, "Processed Row Count": len(X),
             "Processed Feature Count": X.shape[1]}, ignore_index=True)

    if feature == 'memory.max_usage_in_bytes':
        Runtime_Datasets.GENERAL_INFORMATION_MEMORY = Runtime_Datasets.GENERAL_INFORMATION_MEMORY.append(
            {'File Name': Runtime_File_Data.EVALUATED_FILE_NAME, "Test Score": test_score,
             "Train Score": train_score, "Potential Over Fitting": overFitting,
             "Initial Row Count": Runtime_File_Data.EVALUATED_FILE_ROW_COUNT,
             "Initial Feature Count": Runtime_File_Data.EVALUATED_FILE_COLUMN_COUNT, "Processed Row Count": len(X),
             "Processed Feature Count": X.shape[1]}, ignore_index=True)

    value_comparison = value_comparison.assign(y=pd.Series(y_test))
    value_comparison = value_comparison.assign(y_test_hat=pd.Series(y_test_hat))

    # f_regression(X, y)

    # Plot y vs y hat plot
    Plotting_Full_DS.plot(value_comparison, f"{feature}_y_vs_y_hat")
