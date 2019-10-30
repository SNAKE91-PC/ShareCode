'''
Created on Dec 18, 2018

@author: snake91
'''

from mle import simulate as sim
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

p1 = np.asmatrix([
                       [0., 0.],
                       [0., 0.]
                      ])

pMatrix = [p1]

q1 = np.asmatrix([
                    [0., 0.],
                    [0., 0.]
                 ])

qMatrix = [q1]

y0 = np.asmatrix([0., 0]).T

dcoeff = [0,0.5]

t = 500

X = sim.varfimapdqGaussian(t = t, 
                            pMatrix = pMatrix, 
                            qMatrix = qMatrix, 
                            dcoeff = dcoeff, 
                            y0 = y0)


x1 = np.asarray(X[0,:]).reshape(t)
x2 = np.asarray(X[1,:]).reshape(t)

plot_acf(x1, lags = 100)
plot_pacf(x1, lags = 100)
plot_acf(x2, lags = 100)
plot_pacf(x2, lags = 100)
# plot_acf(np.diff(x2, 2), lags = 10)
# plot_pacf(np.diff(x2,2), lags = 10)

plt.plot(x1)
plt.plot(x2)
# plt.plot(np.diff(x2, 2))