from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Lasso, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from Services.PreProcessing import normalize_X, remove_random_rows
import Constants
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


# Negative crossvalidation score
# https://stackoverflow.com/questions/21443865/scikit-learn-cross-validation-negative-values-with-mean-squared-error

# Why splitting twice
# https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn


def predict_memory_usage(df, column_to_remove):
    """
    Trains the model to predict memory usage
    :param df:
    :param column_to_remove
    :return:
    """

    # Prepare dataframe
    y = df[column_to_remove]
    del df[column_to_remove]
    X = df
    X = normalize_X(X)

    # Select model
    model = select_model()

    # Calculate cross validation
    scores = []

    print(np.mean(cross_val_score(model, X, y, cv=5)))
    scores.append(np.mean(cross_val_score(model, X, y, cv=5)))

    return model, np.mean(scores), np.var(scores)


def predict_total_time(df):
    """
    Trains the model to predict the total time
    :param df:
    :return:
    """
    model = select_model()

    y = df['runtime']
    del df['runtime']
    X = df
    X = normalize_X(X)

    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    scores = pd.DataFrame(0, index=np.arange(5),
                          columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '99'])
    print(scores)
    for train_index, test_index in kf.split(X):
        counter = 0
        # print("Train Index: ", train_index, "\n")
        # print("Test Index: ", test_index)

        r2scores = []
        # Iterate from 0 to 101. Ends @ 100, reduce by 1
        for i in range(0, 101, 10):
            if i == 100:
                i = 99
            # Create a deep copy, to keep the original data set untouched
            train_index_copy = train_index

            source_len = len(train_index_copy)

            # Calculate amount of rows to be removed
            rows = int(len(train_index_copy) * i / 100)

            # Remove rows by random index
            train_index_copy = remove_random_rows(train_index_copy, rows)
            print(f"Removed {rows} rows, {len(train_index_copy)} indices left of {source_len}. {100 - i}% data left!")

            X_train, X_test = X[train_index_copy], X[test_index]
            y_train, y_test = y[train_index_copy], y[test_index]

            # print(f"y_test contains {len(y_test)} rows")
            # print(f"X_test contains {len(X_test)} rows")

            # print(f"y_train contains {len(y_train)} rows")
            # print(f"X_train contains {len(X_train)} rows")

            model.fit(X_train, y_train)
            y_test_hat = model.predict(X_test)
            r2scores.append(r2_score(y_test, y_test_hat))

            print(f"R2 score is {r2_score(y_test, y_test_hat)}")

        print(r2scores)
        print(scores.iloc[counter, 1:])
        scores.iloc[counter, 1:] = r2scores
        counter += 1
        print(scores)
    return model


def select_model():
    """
    Select the appropriate model based on the given argument
    """
    if Constants.SELECTED_ALGORITHM == Constants.Model.RIDGE.name:
        return RidgeCV(alphas=[0.1, 1.0, 10.0])

    elif Constants.SELECTED_ALGORITHM == Constants.Model.LASSO.name:
        return Lasso()

    else:
        return RandomForestRegressor(n_estimators=12, random_state=1)
