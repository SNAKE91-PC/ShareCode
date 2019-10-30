'''
Created on Dec 11, 2018

@author: snake91
'''


import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

x = np.random.normal(size = 500)
window = 10

y = [np.mean(x[i-window: i]) for i in range(window, len(x))]


plt.plot(x[window: len(x)])
plt.plot(y)

plot_acf(x, lags = 10)
plot_pacf(x, lags = 10)
plot_acf(y, lags = 10)
plot_pacf(y, lags = 10)