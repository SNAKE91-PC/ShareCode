'''
Created on Dec 18, 2018

@author: snake91
'''

from mle import simulate as sim
from mle import likelihood as logL
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as opt
import scipy.linalg as slin
from copy import deepcopy
import pandas as pd

import mle.constraint as cons

t = 10000


# p1 = np.asmatrix([
#                     [0.1, 0.2], 
#                     [0., 0.1]     
#                 ])
# 
# p2 = np.asmatrix([
#                     [0.1, 0.,], 
#                     [0., 0.1]
#                          
#                 ])
# 
# p3 = np.asmatrix([
#                     [0., 0.3],
#                     [0.1, 0.2]
#     
#                     ])

p1 = np.asmatrix([
                    [0.2, 0., 0.], 
                    [0., 0.2, 0.],
                    [0.4, 0., 0.1]     
                ])
 
p2 = np.asmatrix([
                    [0.1, 0., 0.], 
                    [0.2, -0.2, 0.],
                    [0., 0., 0.1]     
                ])

p = [p1, p2] #, p3]

y0 = np.asmatrix([[0., 0., 0.], [0., 0., 0.]]).T

X = sim.varpGaussian(t =t, pMatrix = p, y0 = y0)

x1 = np.asarray(X[0,:]).reshape(t)
x2 = np.asarray(X[1,:]).reshape(t)
x3 = np.asarray(X[2,:]).reshape(t)



# def constraint(phi, p):
#     
#     nprocess = int(np.sqrt(len(phi)/p))
#     phiList = []
#     
#     phi = list(phi)
# #     eigList = []
#     for i in range(nprocess**2, len(phi)+1, nprocess**2):
#         phiMatrix = np.asmatrix(phi[i-nprocess**2: i]).reshape((nprocess, nprocess))
#         phiList.append(phiMatrix)
#         
#     I = np.diag(np.ones(nprocess * p))
    
# #     def fun(x, I, phiList):
# # 
# # #         tmp = deepcopy(np.asmatrix(I))
# #         I = np.asmatrix(I) * len(phiList)
# #         for lag in reversed(range(0, len(phiList))):
# #             I -= (np.asmatrix(phiList[lag-1]) * (np.float(x)**lag))
# #                     
# #         res = np.linalg.det(I)
# #         
# #         return res
# #     
# # 
# #     sols = np.zeros(len(phiList))
# #     
# #     c = 0
# #     maxIter = 0.
# #     while c < 2:
# #           
# #         startNum = np.random.uniform(low = -1, high = 1)  
# #         y = opt.root(fun, x0 = (startNum,), args = (I, phiList))
# #         
# #         if (np.abs(np.float(y.x)) > 1 and y.message == 'The solution converged.') or (maxIter == 100):
# # 
# #             print 'Constraint reached'
# #             print fun(y.x, I, phiList)
# #             return -1
# # 
# #         
# #         if np.abs(np.float(y.x) - sols).min() > 10e-14 and y.message == 'The solution converged.':
# #             
# #             sols[c] = np.float(y.x)  #.append(np.float(y.x))
# #     
# #             c+=1
# #           
# #         maxIter+=1

        
#     F = np.vstack((np.hstack(phiList), I))
#     zeros = np.zeros(shape = (F.shape[0], F.shape[0]-F.shape[1]))
#     
#     F = np.hstack((F, zeros))
#     
#     a,b = np.linalg.eig(F) # @UnusedVariable
#     
#     f = lambda x: (np.real(x)**2 + np.imag(x)**2) < 1 
#     
#     if all(f(a)):
#         return 1
#     else:
#         print 'constraint'
#         return -1
    


nprocess = X.shape[0]
pLag = len(p)
bounds =tuple( [(-0.99, 0.99) for i in range(0, nprocess ** 2 * pLag)])
# x0 = tuple([0.] * nprocess**2 * pLag)

x0 = []
for lag in range(0, pLag):
    x0 += list(np.diag(np.array([0.1] * nprocess)).flatten())
    
x0 = tuple(x0)
 
paramsX = opt.minimize(logL.maxVARpN, 
                                x0 = x0, 
                                args = X,
                                bounds = bounds,
                                constraints = ({'type': 'ineq', 'fun': lambda phi: cons.consVARp(phi, pLag)}),
                                 
                      )
phiX = []
for i in range(nprocess**2, len(paramsX.x)+1, nprocess**2):
    phiMatrix = np.asmatrix(paramsX.x[i-nprocess**2: i]).reshape((nprocess, nprocess))
    phiX.append(phiMatrix)

print(phiX)
# phiX = np.array(paramsX.x).reshape((nprocess,nprocess))





df = pd.DataFrame({'x1':x1, 'x2':x2}) #, 'x3':x3

print('lag1\n')
print(st.linregress(df['x2'].shift(1).dropna(), df['x1'][1:])) # phi = 0
# print st.linregress(df['x2'].shift(1).dropna(), df['x1'][1:]) # phi = 0

print('lag2\n')
print(st.linregress(df['x2'].shift(2).dropna(), df['x1'][2:])) # phi = 1
# print st.linregress(df['x3'].shift(2).dropna(), df['x2'][2:]) # phi = -1

plt.figure()
plt.scatter(df['x2'].shift(1).dropna(), df['x1'][1:], label ='x2 regress x1', s=3.)
# plt.scatter(df['x3'].shift(1).dropna(), df['x2'][1:], label = 'x3 regress x2', s=3.)
plt.legend()

plt.figure()
plt.scatter(df['x2'].shift(2).dropna(), df['x1'][2:], label ='x2 regress x1', s=3.)
# plt.scatter(df['x3'].shift(2).dropna(), df['x2'][2:], label = 'x3 regress x2', s=3.)
plt.legend()

# plt.figure()
# plt.plot(x1, label = 'x1')
# plt.plot(x2, label = 'x2')
# plt.plot(x3, label = 'x3')
# plt.legend()


