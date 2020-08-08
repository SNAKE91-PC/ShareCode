'''
Created on 1 Aug 2020

@author: snake91
'''


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

from sklearn.decomposition import PCA

np.random.seed(10)

x = np.random.normal(size = (2,1000))
x = x - np.reshape(np.mean(x, axis = 1), (1,2)).T

cov = np.array([[1, 0.99], [0.99, 1]])
chol = np.linalg.cholesky(cov)
xcov = np.dot(x.T, chol)


pca = PCA(n_components=2)
pca.fit(xcov)
Xpca = pca.transform(xcov)


def pcafunc(w, x):
    
#     x = np.reshape(x, (1,2))
    w = np.reshape(w, (1,2))
    v = np.var(np.dot(w, x.T))
    
#     print(v)
    return  -v 



def constraint1(w):
    
    cons1 = np.dot(w, w.T) - 1
    
    return cons1 


cons = ({'type': 'eq', 'fun': lambda w: constraint1(w)})

w = spo.minimize(fun = pcafunc, x0 = (np.random.uniform(), np.random.uniform()), args = xcov, constraints = cons)

w = np.reshape(w.x, (1,2))



plt.scatter(Xpca[:,0], Xpca[:,1])
plt.scatter(xcov[:,0], xcov[:,1])
# plt.hist(xw)

print("")








