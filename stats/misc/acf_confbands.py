'''
Created on Jun 17, 2019

@author: snake91
'''

from statsmodels.tsa.stattools import acf, pacf, ccf

import numpy as np
import matplotlib.pyplot as plt


n = 1000
x = [acf(np.random.normal(size = 10000)) for i in range(n)]


plt.hist(list(map(lambda x: x[1], x)), bins = 200)