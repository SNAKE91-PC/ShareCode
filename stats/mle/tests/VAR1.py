'''
Created on Dec 17, 2018

@author: snake91
'''


from mle import simulate as sim
from mle import likelihood as logL
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as opt
import pandas as pd


t = 10000


p1 = np.asmatrix([
                    [0.2, 0.1, 0.1], 
                    [0.1, 0.2, 0.1],
                    [0.1, 0.1, 0.2]     
                ])
y0 = np.asmatrix([0., 0., 0.]).T

X = sim.var1Gaussian(t =t, pMatrix = p1, y0 = y0)
Y = sim.var1Student(t = t, pMatrix = p1, y0 = y0, df = 5)
# print(X[0, :])
# print(X[1, :])

x1 = np.asarray(X[0,:]).reshape(t)
x2 = np.asarray(X[1,:]).reshape(t)
x3 = np.asarray(X[2,:]).reshape(t)

y1 = np.asarray(Y[0,:]).reshape(t)
y2 = np.asarray(Y[1,:]).reshape(t)
y3 = np.asarray(Y[2,:]).reshape(t)


nprocess = X.shape[0]
bounds =tuple( [(-0.99, 0.99) for i in range(0, nprocess ** 2)])
x0 = tuple([0.] * nprocess**2)


def constraint(y):
    
    nprocess = int(np.sqrt(len(y)))
    y = np.array(y).reshape((nprocess, nprocess))
    eigvalues, eigenvectors = np.linalg.eig(y) # @UnusedVariable
    
    f = lambda x: np.abs(x) < 1
    
    if all(f(eigvalues)):
        return  1
    else:
        return -1
    
    return 

paramsX = opt.minimize(logL.maxVAR1N, 
                                x0 = x0, 
                                args = X,
                                bounds = bounds,
                                constraints = ({'type': 'ineq', 'fun': lambda y: constraint(y)})
                                 
                      )
 
phiX = np.array(paramsX.x).reshape((nprocess,nprocess))

paramsY = opt.minimize(logL.maxVAR1T,
                                x0 = x0,
                                args = Y,
                                bounds = bounds,
                                constraints = ({'type': 'ineq', 'fun': lambda y: constraint(y)})
                        )

phiY = np.array(paramsY.x).reshape((nprocess,nprocess))

# print phiX
print(phiY)





dfX = pd.DataFrame({'x1':x1, 'x2':x2, 'x3':x3})
dfY = pd.DataFrame({'y1':y1, 'y2':y2, 'y3':y3})
# TODO: build cross correlation functions
# print st.linregress(df['x2'].shift().dropna(), df['x1'][1:]) # phi = 1
# print st.linregress(df['x3'].shift().dropna(), df['x2'][1:]) # phi = 1

plt.figure()
plt.scatter(dfX['x2'].shift().dropna(), dfX['x1'][1:], label ='x2 regress x1', s=3.)
plt.scatter(dfX['x3'].shift().dropna(), dfX['x2'][1:], label = 'x3 regress x2', s=3.)
plt.legend()

plt.figure()
plt.scatter(dfY['y2'].shift().dropna(), dfY['y1'][1:], label = 'y2 regress y1', s=3.)
plt.scatter(dfY['y3'].shift().dropna(), dfY['y2'][1:], label = 'y3 regress y2', s=3.)
plt.legend()

# plt.figure()
# plt.plot(x1, label = 'x1')
# plt.plot(x2, label = 'x2')
# plt.plot(x3, label = 'x3')
# plt.legend()




