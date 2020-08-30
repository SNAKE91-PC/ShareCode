'''
Created on 26 Aug 2020

@author: snake91
'''


import numpy as np
import pandas as pd
import scipy.optimize as spo
import itertools as it
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pathos.pools as pp


np.random.seed(10)

x = np.random.normal(size = 2000)/250. #prices
x = np.exp(np.cumsum(x))

horizon = 10 #dd


dev_x = x[:750]
dev_x = pd.Series(dev_x)


def strategy(w, dev_x, horizon):

    w1 = int(w[0])
    w2 = int(w[1])
    
    if w1 == 0 or w2 == 0:
        return (w1, w2, np.nan)
    
    
    strat1 = pd.Series(dev_x).rolling(window = w1).apply(np.mean)
    strat2 = pd.Series(dev_x).rolling(window = w2).apply(np.mean)

    entrypoint = pd.Series(np.where(strat2 > strat1, 1, 0))
    
    data = pd.DataFrame([dev_x, entrypoint, dev_x.shift(horizon)]).transpose()
    
    data.columns = ["dev_x", "entry", "dev_xshift"]

    data = data[data["entry"] == 1]
        
    ret = np.mean( (data["dev_xshift"] - data["dev_x"]) / data["dev_x"] )
    
    print(w1, w2, ret, sep = " ")
    
    return (w1,w2,ret)


# comb = list(it.permutations(np.arange(10, 100, 20), 2))
# comb = list(filter(lambda x: x[0] < x[1], comb))

# cons = {'type': 'ineq', 'fun' : lambda x: x[1] - x[0],
#         'type': 'eq', 'fun' : lambda x: x[1] - int(x[1]),
#         'type': 'eq', 'fun': lambda x: x[0] - int(x[0]) }
#  
# res = spo.minimize(fun = strategy, x0 = (1, 100), args = (dev_x, horizon), bounds = [(0, None), (0, None)], constraints = cons) 
#  
# w = res.x
# print(res.x)



items = np.arange(1,101,1)
comb = [(items[i],items[j]) for i in range(len(items)) for j in range(0, len(items))]



xaxis = np.arange(1,101,1)
yaxis = np.arange(1,101,1)

arr = np.array(np.meshgrid(xaxis, yaxis)).T.reshape(-1,2)
arr = np.array(list(filter(lambda x: x[0] > x[1], arr)))

arr = np.vstack([np.zeros((100-50,2)), arr])

pool = pp.ProcessPool(16)

profit = list(pool.map(lambda x: strategy(x, dev_x, horizon), arr))

Z = list(map(lambda x: x[2], profit))

arr = arr.T.reshape(2,150,300)
X,Y = arr

pool.close()

# Z = np.reshape(np.arange(0,25), (5,5))


fig = plt.figure()

ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');



print("")

