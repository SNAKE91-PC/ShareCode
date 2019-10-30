'''
Created on Apr 11, 2019

@author: snake91
'''


import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
# from sklearn import linear_model, datasets
from statsmodels.regression.linear_model import OLS


def F1(coeff, scale = 1):

    x1 = np.random.normal(size = 100, scale = scale)
    x2 = np.random.normal(size = 100, scale = scale)
    
    X = np.vstack((x1,x2))
    a = coeff[0]
    b = coeff[1]
    
    Y = a * X[0] + b * X[1]
    
    
    regr = OLS(Y, X.T)#linear_model.LinearRegression()
    res = regr.fit()
    
    
    return np.var(res.resid) / np.var(Y)


def ecdf(value, distr):
    
    return len(list(filter(lambda x: value < x, distr)))/ float(len(distr))


x = [F1(coeff = [0.5, 0.7], scale = 0.2) for i in range(1000)]
y = [F1(coeff = [0.2, 0.1], scale = 1.) for i in range(1000)]

pval = [ecdf(i, x) for i in y]


plt.hist(pval)
plt.hist(x, bins = 200)
plt.hist(y, bins = 200)



    




