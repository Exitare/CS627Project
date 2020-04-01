from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from Services import Config, File
from Services.Plotting import Single_Plotting
import pandas as pd
import numpy as np
import Constants
import sys


def compare_real_to_predicted_data(df):
    """
    Creates a plot which contains the real data compared to the predicted values
    :param df:
    :return:
    """
    try:
        if 'runtime' in df.columns:
            value_comparison = pd.DataFrame(columns=['y', 'y_test_hat'])

            model = RandomForestRegressor(n_estimators=Config.Config.FOREST_ESTIMATORS, random_state=1)
            copy = df
            y = copy['runtime']
            del copy['runtime']
            X = copy
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)

            model.fit(X_train, y_train)
            y_test_hat = model.predict(X_test)

            value_comparison = value_comparison.assign(y=pd.Series(y_test))
            value_comparison = value_comparison.assign(y_test_hat=pd.Series(y_test_hat))
            value_comparison = value_comparison.fillna(0)

            Single_Plotting.plot(value_comparison, "y_vs_y_hat")
            # Plotting.plot(value_comparison, Constants.CURRENT_EVALUATED_TOOL_DIRECTORY, 'Value_Comparison_Runtime')

        if 'memory.max_usage_in_bytes' in df.columns:
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
            # Plotting.plot(value_comparison, Constants.CURRENT_EVALUATED_TOOL_DIRECTORY, 'Value_Comparison_Memory')

    except BaseException as ex:
        print(ex)
        File.remove_folder(Constants.CURRENT_WORKING_DIRECTORY)
        sys.exit()
