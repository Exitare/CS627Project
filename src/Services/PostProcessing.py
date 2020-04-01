import Constants


def adjust_r_squared_scores():
    """
    Adjust all negative r2 scores to be 0. r2 score = 0 = mean. r2 < 0 < mean
    :return:
    """
    Constants.RUNTIME_MEAN_REPORT = Constants.RUNTIME_MEAN_REPORT.clip(lower=0)
    Constants.RUNTIME_VAR_REPORT = Constants.RUNTIME_VAR_REPORT.clip(lower=0)
    Constants.MEMORY_VAR_REPORT = Constants.MEMORY_VAR_REPORT.clip(lower=0)
    Constants.MEMORY_MEAN_REPORT = Constants.MEMORY_MEAN_REPORT.clip(lower=0)



