import numpy as np


class activation_functions:
    def __init__(self):
        pass

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def ReLU(x):
        if x >= 0:
            return x
        else:
            return 0

    @staticmethod
    def leakyReLU(x):
        if x >= 0:
            return x
        else:
            return 0.001*x

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def adaline(x):
        return x


