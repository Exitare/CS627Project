class Score:
    def __init__(self, testScore: dict, trainScore: dict, crossValidationScore: dict):
        self.testScore = testScore
        self.trainScore = trainScore
        self.crossValidationScore = crossValidationScore
