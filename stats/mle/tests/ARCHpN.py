'''
Created on Jan 14, 2019

@author: snake91
'''


from mle import simulate as sim
from mle import likelihood as logL
from mle import constraint as cons

import scipy.stats as st
import scipy.optimize as opt
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model

alpha = [0.5, 0.2]
a0 = 0.5

x = sim.archpGaussian(t = 5000, a0 = a0, alpha =  alpha)

p = len(alpha)
x0 = [0. for i in range(p + 1)]
bounds = [(0.01, 0.99) for i in range(p + 1)]

params = opt.minimize(fun = logL.maxARCHpN, 
                        x0 = x0, 
                        bounds = bounds, 
                        args = x, 
                        tol=10e-16, 
                        constraints = ({'type': 'ineq', 'fun': lambda y: cons.consARCHp(y[1:])})
                        )

print(params)
obj = arch_model(x)#, p=1, q=0, o=0)
print(obj.fit()) #check #ok
