class Score:
    def __init__(self, testScore: float, trainScore: float, crossValidationScore: float, variance: int):
        self.testScore = testScore
        self.trainScore = trainScore
        self.crossValidationScore = crossValidationScore
        self.variance = variance
