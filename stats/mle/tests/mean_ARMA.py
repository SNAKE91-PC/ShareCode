'''
Created on 25 Jul 2020

@author: snake91
'''


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

from mle.simulate import armapqGaussian
from mle.likelihood import maxARMApqN


niter = 6000


x = [armapqGaussian(t = 200, phi = [0.], y0 = [0.]) for i in range(niter)]

x = np.array(x)

xavg = np.mean(x, axis = 0)

paramsN = spo.minimize(maxARMApqN, x0 = (0.,), args = (xavg, 1, 0))

distributionParamsN = [spo.minimize(maxARMApqN, x0 = (0.,), args = (x[i], 1, 0)).x[0] for i in range(niter)]

distributionParamsN = np.array(distributionParamsN)

xmd = np.median(x, axis = 0)
xpc_99 = np.percentile(x, 99, axis = 0)




plt.plot(xavg)
plt.plot(xmd)
plt.plot(xpc_99)
print("") 