'''
Created on Jan 17, 2019

@author: snake91
'''



from mle import simulate as sim
from mle import likelihood as logL
import mle.constraint as cons

import scipy.stats as st
import scipy.optimize as opt
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model

alpha = [0.2]
beta = [0.5]
a0 = 0.2
x = sim.garchpqGaussian(t = 5000, a0= a0, alpha =  alpha, beta = beta)

p = len(alpha)
q = len(beta)
x0 = [0. for i in range(p + q + 1)]
bounds = [(0.01, 0.99) for i in range(p + q + 1)]

params = opt.minimize(fun = logL.maxGARCHpqN, 
                        x0 = x0, 
                        bounds = bounds, 
                        args = (x,p,q), 
                        tol=10e-16
                        )


# constraints = (
#                 {'type':'ineq','fun' : lambda y: cons.consARCHp( y[1:][0:p] )},
#                 {'type':'ineq','fun': lambda y: cons.consARCHp( y[1:][p:(p+q)] )}
#               )

print(params)
obj = arch_model(x)#, p=1, q=0, o=0)
print(obj.fit()) #check #ok

