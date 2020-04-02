from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from Services import Config
from Services.File import General
from Services.Plotting import Single_Plotting
import pandas as pd
import RuntimeContants
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
            if test_score < train_score:
                overFitting = True

            RuntimeContants.OVER_UNDER_FITTING = RuntimeContants.OVER_UNDER_FITTING.append(
                {'File Name': f"{RuntimeContants.CURRENT_EVALUATED_FILE}", "Test Score": f"{test_score}",
                 "Train Score": f"{train_score}", "Potential Over Fitting": f"{overFitting}"}, ignore_index=True)
            print(RuntimeContants.OVER_UNDER_FITTING)
            value_comparison = value_comparison.assign(y=pd.Series(y_test))
            value_comparison = value_comparison.assign(y_test_hat=pd.Series(y_test_hat))
            # Na values are set to 0
            value_comparison = value_comparison.fillna(0)

            f_regression(X, y)
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
        General.remove_folder(RuntimeContants.CURRENT_WORKING_DIRECTORY)
        sys.exit()
