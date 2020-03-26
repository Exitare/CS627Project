import Constants


def adjust_r_squared_scores():
    Constants.RUNTIME_MEAN_REPORT = Constants.RUNTIME_MEAN_REPORT.clip(lower=0)
    Constants.RUNTIME_VAR_REPORT = Constants.RUNTIME_VAR_REPORT.clip(lower=0)
    Constants.MEMORY_VAR_REPORT = Constants.MEMORY_VAR_REPORT.clip(lower=0)
    Constants.MEMORY_MEAN_REPORT = Constants.MEMORY_MEAN_REPORT.clip(lower=0)

