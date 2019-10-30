'''
Created on Nov 26, 2018

@author: snake91
'''

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

z = simulate.arma11Gaussian(t = 5000, phi = 0.2, psi = 0.2, y0 = 0)

N = lambda x, mu, sigma : (1./np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2 / (2*sigma**2))
T = lambda x, mu, sigma, df: st.t.pdf(x, df, loc = mu, scale = sigma)


LikelihoodN = [[(phi, psi, -logL.maxARMA11N(params = (phi, psi), x= z)) for phi in np.arange(-0.99, 0.99, 0.1)] for psi in np.arange(-0.99,0.99,0.1)]

############################ 3d plot #
# plt.figure()
# plt.plot(map(lambda x: x[0], LikelihoodN), map(lambda x: x[1], LikelihoodN), label = 'MA1N')
# plt.legend()
# plt.show()
######################################

import scipy.optimize as opt

paramsN = opt.minimize(logL.maxARMA11N, x0 = (0.5, 0.8), args = z)
# paramsN = opt.brute(logL.maxARMA11, ranges = ((-0.99, 0.99), (-0.99, 0.99), (0.1, 2)), args = (z,) )


phiN = paramsN[0]
psiN = paramsN[1]
# sigma = paramsN[2]

print(paramsN)                
                

e0 = 0
eList = [e0]
for t in range(1,len(z)):
    e = z[t] - phiN * eList[t-1] - psi * z[t-1]
    eList.append(e)

eN = eList
plt.figure()
plt.hist(eN, bins = 100, histtype="step")
plt.show()

st.normaltest(eN)
print(st.norm.fit(eN))

# plot_acf(z, lags = 10)
# plot_pacf(z, lags = 10)
