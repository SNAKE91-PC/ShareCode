'''
Created on 9 Aug 2020

@author: snake91

'''


import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import mle.simulate as sim
from sklearn.decomposition import PCA

### AR

lags = 20

phi = [0.999]
x = sim.arpGaussian(t = 5000, phi = phi)
x -= np.mean(x[lags:])
xlags = np.array([pd.Series(x).shift(i)[lags:] for i in range(lags)])


pca = PCA(n_components=lags)
xpca = pca.fit(xlags)

# print(xpca.explained_variance_ratio_)

print(np.cumsum(xpca.explained_variance_ratio_))
autocorr = np.array([pd.Series(x).autocorr(lag = i) for i in range(100)])
covar = xpca.get_covariance()

#### MA

lags = 20
y = sim.maqGaussian(t = 5000, psi = phi)
y -= np.mean(y[lags:])
ylags = np.array([pd.Series(y).shift(i)[lags:] for i in range(lags)])


pca = PCA(n_components=lags)
ypca = pca.fit(ylags)

# print(ypca.explained_variance_ratio_)
print(np.cumsum(ypca.explained_variance_ratio_))
covma = ypca.get_covariance() 

plt.plot(np.cumsum(xpca.explained_variance_ratio_), label = "AR")
plt.plot(np.cumsum(ypca.explained_variance_ratio_), label = "MA")
plt.legend()
plt.show()

print("")







