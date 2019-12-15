from enum import Enum


class Model(Enum):
    LINEAR = 1
    RIDGE = 2
    LASSO = 3


# The selected Algorithm
SELECTED_ALGORITHM = ''
