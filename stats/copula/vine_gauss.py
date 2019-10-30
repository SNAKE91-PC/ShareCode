'''
Created on Mar 11, 2019

@author: snake91
'''


import numpy as np
import scipy.stats as st
import scipy.integrate as scint
import functools
import matplotlib.pyplot as plt

X = np.asmatrix(np.random.normal(size = (3, 1000)))


gauss = lambda u,v, theta: (1/(2*np.pi * np.sqrt(1-theta**2))) * \
                            np.exp( -((u**2 - 2 * theta * u * v + v**2) / (2*(1-theta**2))) )
 
f = lambda u,v,theta: scint.nquad(functools.partial(gauss, theta = theta), [[-np.inf, u], [-np.inf, v]])[0]
    
    
# corrMatrix = np.asmatrix([
#                             [1, 0.2, 0.2],
#                             [0.2, 1, 0.2],
#                             [0.2, 0.2, 1]
#                         ])

corrMatrix1 = np.asmatrix([
                            [1, 0.5],
                            [0.5, 1]
    
                        ])

corrMatrix2 = np.asmatrix([
                            [1, 0.1],
                            [0.1, 1]
                            ]
                        )

chol1 = np.linalg.cholesky(corrMatrix1)

corr2d = np.asarray(chol1 * X[0:2])

corr2dList = list(zip(*corr2d.tolist()))

C = list(map(lambda x: f(x[0], x[1], theta = 0.5), corr2dList))

C = np.array(C)

N = np.asmatrix(st.norm.ppf(C))

Y = np.vstack((N, X[2]))

chol2 = np.linalg.cholesky(corrMatrix2)

corrOuter = chol2 * Y

Final = np.vstack((np.asmatrix(corr2d), corrOuter[1]))

print('')
















