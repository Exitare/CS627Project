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
    memoryScore = Score(0, 0, 0, 0)

    if 'memtotal' in df.columns:
        model, crossScore, variance = predict_memory_usage(df, 'memtotal')
        memoryScore.crossValidationScore = crossScore
        memoryScore.variance = variance

    if 'memory.max_usage_in_bytes' in df.columns:
        model, crossScore, variance = predict_memory_usage(df, 'memory.max_usage_in_bytes')
        memoryScore.crossValidationScore = crossScore
        memoryScore.variance = variance

    return memoryScore


def calculate_runtime(df, percent):
    rows = int(len(df.index) * percent / 100)
    df = remove_random_rows(df, rows)

    if 'runtime' in df.columns:
        runtimeScore = Score(0, 0, 0, 0)
        model, crossScore, variance = predict_total_time(df)
        runtimeScore.crossValidationScore = crossScore
        runtimeScore.variance = variance
        return runtimeScore

    return None
