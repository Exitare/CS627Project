from Services.PreProcessing import remove_random_rows
from Services.Predictions import predict_memory_usage, predict_total_time
from Entities.Evaluations import Score
import numpy as np


def calculate_row_by_row(df):
    deepCopy = df
    for i in range(len(deepCopy.index)):
        print("")
        df = deepCopy
        df = remove_random_rows(df, i)

        memoryScores = Score({"value": 0, "rowcount": 0}, {"value": 0, "rowcount": 0},
                             {"value": 0, "rowcount": 0})
        runtimeScores = Score({"value": 0, "rowcount": 0}, {"value": 0, "rowcount": 0},
                              {"value": 0, "rowcount": 0})

        print("-------------")
        print(i)
        print(f"Test with {len(df.index)} rows")
        print("")
        # Total memory usage
        if 'memtotal' in df.columns:
            model, testScore, trainScore, crossScore = predict_memory_usage(df)

            crossScore = np.mean(crossScore)

            if float(testScore) >= memoryScores.testScore.get("value"):
                memoryScores.testScore = {"value": testScore, "rowcount": len(df.index)}

            if float(trainScore) >= memoryScores.trainScore.get("value"):
                memoryScores.trainScore = {"value": trainScore, "rowcount": len(df.index)}

            if float(crossScore) >= memoryScores.crossValidationScore.get("value"):
                memoryScores.crossValidationScore = {"value": crossScore, "rowcount": len(df.index)}

        #  if Constants.SELECTED_ALGORITHM != Constants.Model.FOREST.name:
        #     stats.print_coef(df, model)

        # If model selection is Ridge print best alpha value
        # if Constants.SELECTED_ALGORITHM == Constants.Model.RIDGE.name:
        #   stats.print_alpha(model)

        # Total runtime
        if 'runtime' in df.columns:
            print("Predicting total runtime")
            model, testScore, trainScore, crossScore = predict_total_time(df)
            crossScore = np.mean(crossScore)

            if float(testScore) >= runtimeScores.testScore.get("value"):
                runtimeScores.testScore = {"value": testScore, "rowcount": len(df.index)}

            if float(trainScore) >= runtimeScores.trainScore.get("value"):
                runtimeScores.trainScore = {"value": trainScore, "rowcount": len(df.index)}

            if float(crossScore) >= runtimeScores.crossValidationScore.get("value"):
                runtimeScores.crossValidationScore = {"value": crossScore, "rowcount": len(df.index)}

        # if Constants.SELECTED_ALGORITHM != Constants.Model.FOREST.name:
        #    stats.print_coef(df, model)

        # If model selection is Ridge print best alpha value
        # if Constants.SELECTED_ALGORITHM == Constants.Model.RIDGE.name:
        #   stats.print_alpha(model)


def calculate_memory(df, percent):
    """

    :param df: The dataframe
    :param percent: The amount of rows which are getting removed
    :return:
    """

    rows = int(len(df.index) * percent / 100)
    df = remove_random_rows(df, rows)
    print(f"Remaining row count {len(df.index)}")

    memoryScore = Score(0, 0, 0)

    if 'memtotal' in df.columns:
        model, testScore, trainScore, crossScore = predict_memory_usage(df)
        memoryScore.trainScore = trainScore
        memoryScore.testScore = testScore
        memoryScore.crossValidationScore = crossScore

    return memoryScore


def calculate_runtime(df, percent):
    rows = int(len(df.index) * percent / 100)
    df = remove_random_rows(df, rows)
    print(f"Remaining row count {len(df.index)}")

    runtimeScore = Score(0, 0, 0)

    if 'runtime' in df.columns:
        model, testScore, trainScore, crossScore = predict_total_time(df)
        runtimeScore.testScore = testScore
        runtimeScore.trainScore = trainScore
        runtimeScore.crossValidationScore = crossScore

    return runtimeScore
