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

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for i in range(1, 99, 10):
            print(i)

            rows = int(len(X_train) * i / 100)
            print(rows)
            X_train = remove_random_rows(X_train, rows)

            print("fitting")
            model.fit(X_train, y_train)

            y_test_hat = model.predict(X_test)
            print(r2_score(y, y_test_hat))

    return model


def splitting_model(X, y):
    """
    Using the train_test_split function twice, to generate a valid train, test and validation set.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    return X_train, X_test, y_train, y_test


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
