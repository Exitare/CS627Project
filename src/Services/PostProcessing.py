from RuntimeContants import Runtime_Datasets


def adjust_r_squared_scores():
    """
    Adjust all negative r2 scores to be 0. r2 score = 0 = mean. r2 < 0 < mean
    :return:
    """
    Runtime_Datasets.RUNTIME_MEAN_REPORT = Runtime_Datasets.RUNTIME_MEAN_REPORT.clip(lower=0)
    Runtime_Datasets.RUNTIME_VAR_REPORT = Runtime_Datasets.RUNTIME_VAR_REPORT.clip(lower=0)
    Runtime_Datasets.MEMORY_VAR_REPORT = Runtime_Datasets.MEMORY_VAR_REPORT.clip(lower=0)
    Runtime_Datasets.MEMORY_MEAN_REPORT = Runtime_Datasets.MEMORY_MEAN_REPORT.clip(lower=0)



