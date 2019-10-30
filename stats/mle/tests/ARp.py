'''
Created on Dec 8, 2018

@author: snake91
'''


from mle import simulate as sim
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import numpy as np
from mle import likelihood as logL
import scipy.optimize as opt
import mle.constraint as cons

import datetime

np.random.seed(10)
phi = [0.1, 0.2, 0.1]

x = sim.arpGaussian(t = 500, phi = phi)

# sigma = 1
x0 = tuple([0. for i in range(len(phi))])# + [sigma])              
bounds = [(-0.99, 0.99) for i in range(len(phi))]# + [(0, None)]     

    
# plot_acf(x, lags = 10)
# plot_pacf(x, lags = 10)

time1 = datetime.datetime.now()

# x0 = tuple([0. for i in range(len(phi))])
# bounds = [(-0.99, 0.99) for i in range(len(phi))]

# params = opt.minimize(logL.maxARpN, x0 = x0, \
#                         args = x, \
#                         bounds = bounds,
#                         constraints= ({'type': 'ineq', 'fun': lambda y: cons.consARp(y)})#, \
#                         
#                       )

params = opt.minimize(logL.maxARpN, x0 = x0, \
                        args = x, \
                        bounds = bounds,
                        constraints= ({'type': 'ineq', 'fun': lambda y: cons.consARp(y)})#, \
                        
                      )
    
time2 = datetime.datetime.now()

print(time2 - time1)
# constraints= ({'type': 'ineq', 'fun': lambda y: np.sum(np.abs(y[:-1])) - 1})
print(params)

