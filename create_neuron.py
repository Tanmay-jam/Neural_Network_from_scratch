from typing import List

from NN_objects import Neuron
from activation import activation_functions

'''create a single neuron
with 4 inputs and bias=1. Use sigmoid as activation function.
choices for activation function are: sigmoid, ReLU, leakyReLU, tanh, adaline
'''

neuron_1 = Neuron(no_ip=4, actv="sigmoid", bias=1)

'''setting weights of neuron.
NOTE: dimension of weights list should match with dimension of inputs. As there will be same number of weighted inputs
as the number of inputs to neuron ( considering bias has given some default weight)'''
wght: list[float] = [0.5, 0.56, 0.32, 0.1]
neuron_1.set_weights(wght)

'''activate the neuron by giving input to the neuron and passing sum of weighted input to activation function of the neuron'''

#Input 1
inpt: list[float] = [2, 4, 3.5, 0]
output = neuron_1.run_neuron(inpt)
print(output)

#Input 2
inpt2: list[float] = [0.5, -0.989, -1, 4]
output2 = neuron_1.run_neuron(inpt2)
print(output2)