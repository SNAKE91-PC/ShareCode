'''
Created on Jan 5, 2019

@author: snake91
'''

import numpy as np
import scipy.linalg as slin
import pandas as pd
import matplotlib.pyplot as plt

from mle import simulate as sim
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf


np.random.seed(1)
x = sim.ar1Gaussian(t = 500, phi = 0.5)

phi = 0.5

acfList = [phi ** n for n in range(500)]

np.random.seed(1)
y = np.random.normal(size = 500)

z = slin.toeplitz(acfList)


zinv = np.linalg.cholesky(z)

zinv = np.asmatrix(zinv)
y = np.asmatrix(y)

r = y * zinv

plt.plot(acf(x), label = 'classical way')
plt.plot(acf(r), label = 'cholesky')
plt.legend()


