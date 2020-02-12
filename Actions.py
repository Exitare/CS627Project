from Services.PreProcessing import remove_random_rows
from Services.Predictions import predict_memory_usage, predict_total_time
from Entities.Evaluations import Score
import numpy as np


def calculate_memory(df, percent):
    """

    :param df: The dataframe
    :param percent: The amount of rows which are getting removed
    :return:
    """

    rows = int(len(df.index) * percent / 100)
    df = remove_random_rows(df, rows)
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
    runtimeScore = Score(0, 0, 0)

    if 'runtime' in df.columns:
        model, testScore, trainScore, crossScore = predict_total_time(df)
        runtimeScore.testScore = testScore
        runtimeScore.trainScore = trainScore
        runtimeScore.crossValidationScore = crossScore

    return runtimeScore
