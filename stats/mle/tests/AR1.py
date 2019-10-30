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

x = simulate.ar1Student(t = 500, phi = 0.5, df = 1) #, var, unvar
y = simulate.ar1Gaussian(t = 500, phi = 0.5) #, var, unvar 
z = simulate.ma1Gaussian(t = 500, phi = 0.5)

N = lambda x, mu, sigma : (1./np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2 / (2*sigma**2))
T = lambda x, mu, sigma, df: st.t.pdf(x, df, loc = mu, scale = sigma)

sigma = 1


    
import scipy.optimize as opt

paramsAR1T = opt.minimize(logL.maxAR1T, x0 = (0.1,), args = x, bounds = ((-1, 1),))
paramsAR1N = opt.minimize(logL.maxAR1N, x0 = (0.1,), args = y, bounds = ((-1, 1),))


phiN = paramsAR1T.x#list(reversed(sorted(LikelihoodAR1N, key = lambda x: x[1])))[0][0]
phiT = paramsAR1N.x#list(reversed(sorted(LikelihoodAR1T, key = lambda x: x[1])))[0][0]

eN = [y[t] - phiN * y[t-1] for t in range(1, len(y))]
eT = [x[t] - phiT * x[t-1] for t in range(len(x))]

plt.figure()
plt.hist(eN, bins = 100, histtype="step")
plt.hist(eT, bins = 100, histtype="step")
plt.show()

st.normaltest(eN)
print(st.t.fit(eT))
print(st.norm.fit(eN))

plot_acf(x, lags = 10)
plot_pacf(x, lags = 10)
# plt.plot(var)
# plt.plot(unvar)
# plt.ylim(ymax = max(x)*1.5, ymin = min(var) * 1.5)


# breachvar = filter(lambda x: x<0, [x[i] - var[i] for i in range(len(unvar))])
# breachunvar = filter(lambda x: x<0, [x[i] - unvar[i] for i in range(len(unvar))])

# print len(breachvar)
# print len(breachunvar)