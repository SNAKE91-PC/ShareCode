'''
Created on Nov 18, 2018

@author: snake91
'''

from mle import simulate
from mle import likelihood as logL 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from matplotlib.pyplot import legend

z = simulate.ma1Gaussian(t = 500, psi = 0.8)

N = lambda x, mu, sigma : (1./np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2 / (2*sigma**2))
T = lambda x, mu, sigma, df: st.t.pdf(x, df, loc = mu, scale = sigma)


# func = T
# sigma = 1

# LikelihoodMA1N = []
# LMA1N = []
# 
# for i in np.arange(-0.5, 0.9, 0.01):
#     LMA1N = []
#     e0 = 0
#     eList = [e0]
#     for t in range(1, len(z)):
#         
#         e = z[t] - i * eList[t-1] 
#         eList.append(e)
#         
#         pMA1N = N(z[t], i * eList[t-1], sigma)
#         LMA1N.append(np.log(pMA1N))
#         
#     LMA1N = sum(LMA1N)
#     LikelihoodMA1N.append((i, LMA1N))
    
    
# LikelihoodN = [(phi, -logL.maxMA1N(params = (phi, sigma), x= z)) for phi in np.arange(-1,1,0.1)]
# 
# plt.figure()
# plt.plot(map(lambda x: x[0], LikelihoodN), map(lambda x: x[1], LikelihoodN), label = 'MA1N')
# plt.legend()
# plt.show()

import scipy.optimize as opt

paramsN = opt.minimize(logL.maxMA1N, x0 = (0.1,), args = z, bounds = ((-1, 1),))
psiN = paramsN.x[0]
sigma = paramsN.x[1]

print(paramsN)
# phiMA1N = list(reversed(sorted(LikelihoodMA1N, key = lambda x: x[1])))[0][0]
# print phiMA1N


e0 = 0
eList = [e0]
for t in range(1,len(z)):
    e = z[t] - psiN * eList[t-1]
    eList.append(e)

eN = eList
plt.figure()
plt.hist(eN, bins = 100, histtype="step")
plt.show()

st.normaltest(eN)
print(st.norm.fit(eN))

# plot_acf(z, lags = 10)
# plot_pacf(z, lags = 10)
