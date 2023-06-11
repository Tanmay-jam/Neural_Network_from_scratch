import numpy as np
from activation import activation_functions


class Neuron(activation_functions):
    def __init__(self, no_ip, actv, bias=0):
        self.no_ip = no_ip
        self.bias = bias
        self.actv = actv
        self.weights = np.empty((1, no_ip))

    def set_weights(self, weights):
        self.weights = np.array(weights).reshape((1, self.no_ip))

    def run_neuron(self, n_input):
        assert isinstance(n_input, list)
        n_input = np.array(n_input).reshape((1, self.no_ip))
        print("input:", n_input)
        n_sum = (n_input*self.weights).sum() + self.bias
        print("weighted sum:", n_sum)
        if self.actv == "ReLU":
            n_output = activation_functions.ReLU(n_sum)
        elif self.actv == "leaky ReLU":
            n_output = activation_functions.leakyReLU(n_sum)
        elif self.actv == "sigmoid":
            n_output = activation_functions.sigmoid(n_sum)
        elif self.actv == "tanh":
            n_output = activation_functions.tanh(n_sum)
        elif self.actv == "adaline":
            n_output = activation_functions.adaline(n_sum)

        return n_output




