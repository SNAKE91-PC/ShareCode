'''
Created on 26 Jun 2020

@author: snake91
'''


import matplotlib.pyplot as plt
import numpy as np

### MODEL : Y = sigma(W1 x + b) 
## x : 3x1
## W1: 3x3
## b : 3x1
## Y : 1x1 


# activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)


shape0 = 1000
shape1 = 3

X_file = np.random.normal(size = (shape0, shape1), scale = 0.5)
Y = np.sum(X_file, axis = 1) + np.random.normal(size = shape0, scale =  0.2)

Y = sigmoid(Y)
Y = np.reshape(Y, (shape0, 1))


X_file = np.hstack([Y, X_file])

N = np.shape(X_file)[0]
X = X_file[:,1:4]
# X = np.hstack((np.ones(N).reshape(N, 1), X_file[:, 4].reshape(N, 1)))
# Y = X_file[:, 0] 

w = np.array([1.5, 1.5, 1.5])

# Start batch gradient descent, it will run for max_iter epochs and have a step
# size eta
max_iter = 500
eta = 1E-3

wList = np.zeros(shape = (1,3))
for t in range(0, max_iter):
    
    # We need to iterate over each data point for one epoch
    grad_t = np.array([0., 0., 0.])
    for i in range(0, N):
        x_i = X[i, :]
        y_i = Y[i]

        output = sigmoid(np.dot(w, x_i)) #feedforward
        
        h = 2 * (output - y_i) * np.dot(x_i, sigmoid_derivative(np.dot(w, x_i))) 
        grad_t += x_i *  h   #backpropagation

#         print(grad_t)
    # Update the weights
    print(w)
    
    w = w - eta*grad_t

    if np.sum((wList - w)**2) > 10e-8:
    
        wList[0] = w
        
    else:
        break
    
    
    
print("Weights found:", w)

w = np.reshape(w, (1,3))
X = np.reshape(X, (shape0,3))
Ynew= sigmoid(np.dot(X, w.T))

plt.scatter(Ynew.T, Y)

# plt.scatter(Ynew.T, X[:,0])
# plt.scatter(Y, X[:,0])

plt.show()







