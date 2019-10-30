'''
Created on Nov 29, 2018

@author: snake91
'''

import mle.simulate as sim
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import scipy.stats as st

x = sim.arfimapdqGaussian(t = 5000, phi=[0], psi=[0.], dcoeff=0.7, y0=[0.])



plot_acf(x)
plot_pacf(x)
 
plt.plot(x)



