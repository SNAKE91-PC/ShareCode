'''
Created on 16 Jan 2020

@author: snake91
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


nSim = 1000
nboot = 100
ntrial = 1000

loc = 0
scale = 2
x = np.random.normal(size = nSim, loc = 0, scale = scale)
x = (x - np.mean(x)) / np.sqrt(np.var(x)) * np.sqrt(scale) + loc

#mean
mean_sample = [np.mean(np.random.choice(x, nboot)) for i in range(10000)]
sns.distplot(mean_sample, fit=st.norm, kde=False)


#variance biased 
varbiased_sample = [np.var(np.random.choice(x, nboot)) for i in range(10000)]
sns.distplot(varbiased_sample, fit = st.norm, kde = False)

#variance unbiased
varunbiased_sample = [np.var(np.random.choice(x, nboot), ddof = 1) for i in range(10000)]
sns.distplot(varunbiased_sample, fit = st.norm, kde = False)


#variance bias wrt number of obs
xaxis = np.linspace(10, 1000, 100, dtype = np.int)
bias = [(i, np.mean([np.var(np.random.choice(x, nSim)) for i in range(ntrial)]))] 
yaxis = list(map(lambda x: x[2], bias))


plt.plot(xaxis, yaxis, label = 'biased $\sigma^2$')
plt.plot(xaxis, [2]*len(xaxis), color = 'red', linewidth = 2, linestyle = '--', label = 'theoretical $\sigma^2$')
plt.legend()
plt.xlabel('Sample size')
plt.ylabel('$\sigma^2$')

plt.savefig('/home/snake91/git/ShareCode/stats/bootstrap/pics/bias_variance.svg')

print("")