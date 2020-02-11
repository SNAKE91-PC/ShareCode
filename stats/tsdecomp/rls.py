'''
Created on 28 Jan 2020

@author: snake91
'''

    
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt


np.random.seed(10)

def ols(params, y, x):
     
    weight = list(reversed([0.98**n for n in range(len(x))]))
    predX = weight * params * x
        
    return np.mean((y - predX)**2)




n = 100000
params = [0.5]
x = np.random.normal(size = n)
y = params[0] * x + np.random.normal(size = n)

Params = []
for i in range(1, len(x)):
    
    x0 = (0.)
    bounds = [(None, None)]
    params = spo.minimize(fun = ols, x0 = x0, bounds = bounds, args = (x[:i], y[:i]))
    
    print(i)
    Params.append(params.x[0])
    

# plt.plot(Params)
plt.plot(y)
plt.plot(np.array(Params) * x[1:])
print("")
    
    
    