from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from dataprocessing import remove_bad_columns


def predict_cpu_usage(df):
    """
    Trains the model to predict the cpu usage
    This function will be deprected in further version and is therefore not maintained anymore.
    As galaxy is only setting the cpu count by a handwritten algorithm this is not useful to predict.
    :param df:
    :return:
    """
    print("CPU usage predication started...")

    y = df['processor_count']
    del df['processor_count']
    X = df

    print("Training model...")

    model = Ridge()
    X_train, X_test, y_train, y_test, X_val, y_val = splitting_model(X, y)
    model.fit(X_train, y_train)
    y_test_hat = model.predict(X_test)

    print(f"CPU model test score is : {model.score(X_test, y_test)}")
    print(f"CPU model train score is : {model.score(X_train, y_train)}")
    print(f"Predication: {y_test_hat[:5]}")
    print(f"Y Values: {y_test[:5]}")

    scores = cross_val_score(model, X, y, cv=5)
    print(f"CPU Cross validation score is : {scores}")


def predict_memory_usage(df):
    """
    Trains the model to predict memory usage
    :param df:
    :return:
    """

    y = df['memtotal']
    del df['memtotal']
    X = df

    # model = Ridge(alpha=1.0)
    model = LinearRegression()
    X_train, X_test, y_train, y_test, X_val, y_val = splitting_model(X, y)
    model.fit(X_train, y_train)
    y_test_hat = model.predict(X_train)
    print(f"Memory model test score is : {model.score(X_test, y_test)}")
    print(f"Memory model train score is : {model.score(X_train, y_train)}")
    print(f"Predication: {y_test_hat}")

    scores = cross_val_score(model, X, y, cv=5)
    print(f"Memory Cross validation score is : {scores}")
    print("")
    print(model.coef_)
    print("")


def predict_total_time(df):
    """
    Trains the model to predict the total time
    :param df:
    :return:
    """
    print("Total time predication started...")

    y = df['runtime']
    del df['runtime']
    X = df

    # model = Ridge(alpha=1.0)
    model = LinearRegression()
    X_train, X_test, y_train, y_test, X_val, y_val = splitting_model(X, y)
    model.fit(X_train, y_train)
    y_test_hat = model.predict(X_train)
    print(f"Total time model test score is : {model.score(X_test, y_test)}")
    print(f"Total time model train score is : {model.score(X_train, y_train)}")
    print(f"Predication: {y_test_hat}")

    scores = cross_val_score(model, X, y, cv=5)
    print(f"Total time Cross validation score is : {scores}")


def splitting_model(X, y):
    """
    Using the train_test_split function twice, to generate a valid train, test and validation set.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=1)
    return X_train, X_test, y_train, y_test, X_val, y_val

# https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
