from sklearn.metrics import r2_score
import pandas as pd


def print_additions_stats(df, model):
    print_coef(df, model)
    print_alpha(model)


def printRSquared(y, y_hat):
    print(r2_score(y, y_hat,
                   multioutput='variance_weighted'))


def print_coef(df, model):
    coef = pd.DataFrame()
    coef['Name'] = df.columns
    coef['coef'] = model.coef_
    print("Feature weights:")
    print(coef)
    print("")


def print_alpha(model):
    print("Best alpha value for Ridge:")
    print(model.alpha_)
    print("")


def detect_best_accuracy():
    pass