'''
Created on Dec 18, 2018

@author: snake91
'''


import mle.simulate as sim
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

x = sim.arimapdqGaussian(t = 200, phi = [0.], psi = [0.], dcoeff = 2, y0 = [0.])

plot_acf(x, lags = 10)
plot_acf(np.diff(x, 2), lags = 10)