import RuntimeContants


def adjust_r_squared_scores():
    """
    Adjust all negative r2 scores to be 0. r2 score = 0 = mean. r2 < 0 < mean
    :return:
    """
    RuntimeContants.RUNTIME_MEAN_REPORT = RuntimeContants.RUNTIME_MEAN_REPORT.clip(lower=0)
    RuntimeContants.RUNTIME_VAR_REPORT = RuntimeContants.RUNTIME_VAR_REPORT.clip(lower=0)
    RuntimeContants.MEMORY_VAR_REPORT = RuntimeContants.MEMORY_VAR_REPORT.clip(lower=0)
    RuntimeContants.MEMORY_MEAN_REPORT = RuntimeContants.MEMORY_MEAN_REPORT.clip(lower=0)



