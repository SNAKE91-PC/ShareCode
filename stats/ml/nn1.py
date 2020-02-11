'''
Created on 18 Jan 2020

@author: snake91
'''

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)
#         self.series = np.array([])

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

#         if self.series.shape == (0,):
#             self.series = self.output
#         else:
#             self.series = np.hstack([self.series, self.output])
    
    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        # it should be the same as finding min(y-y^) wrt weights (y^ = self.layer1)
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == "__main__":
    # 3 regressors 4 outputs
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    nn = NeuralNetwork(X,y)

    for i in range(20000):
        nn.feedforward()
        nn.backprop()

    print(nn.output)
    
#     for i in range(0, len(nn.series)):
#         plt.plot(nn.series[i], label = 'series' + str(i))
#     
#         plt.legend()
    
    
    
    
    
    
    
    
    
    