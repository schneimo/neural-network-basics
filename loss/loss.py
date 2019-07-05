import numpy as np


class CrossEntropyLossWithLogits:

    def __init__(self, softmax):
        self.softmax = softmax

    def calc(self, p):
        self.y = p * np.log(self.softmax.y)
        return -1 * np.sum(self.y)

    def derivative(self, p):
        return self.y - p


class MeanSquaredErrorLoss:

    def calc(self, y):
        return
