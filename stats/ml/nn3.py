'''
Created on 11 Jul 2020

@author: snake91
'''


import matplotlib.pyplot as plt
import numpy as np

### MODEL : Y = W2sigma(W1 x + b) 
## x : 3x1
## W1: 3x3
## b : 3x1
## Y : 1x1 
## z : (W1.T x + b).T : 1x3
## W2: 3x1


# activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)


shape0 = 1000
shape1 = 2

X_file = np.random.normal(size = (shape0, shape1), scale = 0.5)

W1 = np.random.normal(size = (shape1, shape1))#np.array([[1., 1.5, 1.], [1, 2, 1], [1, 3, 1]])
W2 = np.random.normal(size = (1, shape1))#np.array([[0.5, 1.5, 2.]])

layer1 = sigmoid(np.dot(W1, X_file.T))
Y = sigmoid(np.dot(W2, layer1))#np.sum(X_file, axis = 1) #+ np.random.normal(size = shape0, scale =  0.2)


Y = np.reshape(Y, (shape0, 1))


X_file = np.hstack([Y, X_file])

N = np.shape(X_file)[0]
X = X_file[:,1:shape1 + 1]
# X = np.hstack((np.ones(N).reshape(N, 1), X_file[:, 4].reshape(N, 1)))
# Y = X_file[:, 0] 

w1 = np.random.uniform(size = (shape1,shape1))#np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
w2 = np.random.uniform(size = (1,shape1))#np.array([[1.5, 1.5, 1.5]])
# Start batch gradient descent, it will run for max_iter epochs and have a step
# size eta
max_iter = 5000
eta = 1E-3

wList1 = np.zeros(shape = (shape1,shape1))
wList2 = np.zeros(shape = (1,shape1))
for t in range(0, max_iter):
    
    # We need to iterate over each data point for one epoch
    grad1 = np.zeros((shape1, shape1))
    grad2 = np.array((1, shape1))#[[0., 0., 0.]])
    for i in range(0, N):
        x_i = np.reshape(X[i, :], (1,shape1))
        y_i = Y[i]

        layer1 = sigmoid(np.dot(w1, x_i.T)).T
        output = sigmoid(np.dot(w2, layer1.T)) #feedforward
        
        d_weights2 = np.dot(layer1.T, 2*(y_i -output)*sigmoid_derivative(output))
        d_weights1 = np.dot(x_i.T, np.dot(2*(y_i -output)*sigmoid_derivative(output), w2)*sigmoid_derivative(layer1))
        
        
#         d_weights2 = np.dot(layer1, 2 * (output - y_i) * sigmoid_derivative(np.dot(w2.T, layer1))) 
        w2 += d_weights2.T #x_i *   #backpropagation
# 
# #         d_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
#         
#         d_weights1 = 2 * (output - y_i) * sigmoid_derivative(np.dot(w2, layer1.T)) * w2 * np.dot(sigmoid_derivative(layer1), x_i)
        w1 += d_weights1.T # x_i

#         print(grad_t)
    # Update the weights
    print(np.sum(output - Y)**2)
    
#     w1 = w1 - eta*grad1
#     w2 = w2 - eta*grad2

#     if np.sum((wList1 - w1)**2) > 10e-8 or np.sum((wList2 - w2)**2) > 10e-8:
#     
#         wList1 = w1
#         wList2 = w2
#     else:
#         break
    
    
    
print("Weights found:", w1)
print("Weights found:", w2)


print("")
# w = np.reshape(w1, (1,3))
# X = np.reshape(X, (shape0,3))
# Ynew= sigmoid(np.dot(X, w.T))
# 
plt.scatter(output, Y)
# 
# plt.scatter(Ynew.T, X[:,0])
# plt.scatter(Y, X[:,0])
# 
plt.show()







