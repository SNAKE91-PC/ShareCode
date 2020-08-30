'''
Created on 21 Aug 2020

@author: snake91
'''


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import statsmodels.formula.api as smf
import pandas as pd

import tensorflow as tf

sample = 500
weights = 5

X = np.random.normal(size = (sample, weights))
w = np.random.uniform(size = (1,weights))

r = np.dot(X, w.T)

reslin = LinearRegression()
res = reslin.fit(X, r)

print(res.coef_ - w)

df = pd.DataFrame(np.hstack([X,r]), columns = ["X1", "X2", "X3", "X4", "X5", "r"])
resq = smf.quantreg('r ~ X1 +X2 + X3 + X4 + X5 + 0', df).fit(q=0.9)
resq.params
