from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from Services.Configuration.Config import Config
from Services.Processing import PreProcessing
import logging


def predict(label: str, dataframe):
    model = RandomForestRegressor(n_estimators=Config.FOREST_ESTIMATORS, max_depth=Config.FOREST_MAX_DEPTH,
                                  random_state=1)

    y = dataframe[label]
    del dataframe[label]
    X = dataframe

    source_row_count = len(X)

    X_indices = (X != 0).any(axis=1)
    X = X.loc[X_indices]
    y = y.loc[X_indices]

    if source_row_count != len(X) and Config.VERBOSE:
        logging.info(f"Removed {source_row_count - len(X)} row(s). Source had {source_row_count}.")

    X = PreProcessing.variance_selection(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

    model.fit(X_train, y_train)

    y_test_hat = model.predict(X_test)
    y_train_hat = model.predict(X_train)
    train_score = r2_score(y_train, y_train_hat)
    test_score = r2_score(y_test, y_test_hat)

    over_fitting = False
    if train_score > test_score * 2:
        over_fitting = True

    return model, train_score, test_score, over_fitting, X, y_test, y_test_hat
