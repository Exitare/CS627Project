from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from dataprocessing import remove_bad_columns


def predict_cpu_usage(df):
    """
    Trains the model to predict the cpu usage
    :param df:
    :return:
    """
    print("CPU usage predication started...")
    data = remove_bad_columns(df)

    y = data['processor_count']
    del data['processor_count']
    X = data

    print("Training model...")

    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
    model.fit(X_train, y_train)
    y_test_hat = model.predict(X_test)

    print(f"CPU model test score is : {model.score(X_test, y_test)}")
    print(f"CPU model train score is : {model.score(X_train, y_train)}")
    print(f"Predication: {y_test_hat[:5]}")
    print(f"Y Values: {y_test[:5]}")

    scores = cross_val_score(model, X, y, cv=5)
    print(f"CPU Cross validation score is : {scores}")
    print(model.coef_)
    # show_plot(X_train, y_train, "square")


# hyper_parameter(X, y)
# Lasso, Ridge
# Grid search
# train data split again, validation data using to set hyperparamter

def predict_memory_usage(df):
    """
    Trains the model to predict memory usage
    :param df:
    :return:
    """
    print("Memory usage predication started...")
    data = remove_bad_columns(df)

    y = data['memtotal']
    del data['memtotal']
    X = data

    model = LinearRegression()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
    model.fit(X_train, y_train)
    y_test_hat = model.predict(X_train)
    print(f"Memory model test score is : {model.score(X_test, y_test)}")
    print(f"Memory model train score is : {model.score(X_train, y_train)}")
    print(f"Predication: {y_test_hat}")

    scores = cross_val_score(model, X, y, cv=5)
    print(f"Memory Cross validation score is : {scores}")
    # show_plot(y_test, y_test_hat, "square")


def hyper_parameter(X, y):
    ridge = make_pipeline(LinearRegression())
    print(ridge.get_params())

    hyperparameters = [{'linearfeatures__degree': range(1, 10)}]
    # the keywords have to match keys in pipe.get_params().keys()

    hyp = GridSearchCV(ridge, hyperparameters,
                       cv=StratifiedKFold(3, shuffle=True),
                       n_jobs=-1)  # parallelize

    result = cross_validate(hyp, X, y,
                            cv=StratifiedKFold(4, shuffle=True),
                            return_estimator=True)

    print(result['test_score'], [est.best_params_ for est in result['estimator']])



# X_train, X_test, y_train, y_test
 #   = train_test_split(X, y, test_size=0.2, random_state=1)

 #X_train, X_val, y_train, y_val
  #  = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


  #https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn