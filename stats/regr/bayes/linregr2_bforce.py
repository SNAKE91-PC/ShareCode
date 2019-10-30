'''
Created on Aug 18, 2019

@author: snake91
'''

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd

b0 = 0.2
b1 = 0.5
size = 100
sigma = 1
x = np.random.normal(size = size, scale = sigma)

y = b0 + b1 * x

#### bayes

likelihood = lambda x, loc, scale: st.norm.pdf(x, loc, scale)
b0_prior = lambda x, loc, scale: st.norm.pdf(x, loc, scale)
b1_prior = lambda x, loc, scale: st.norm.pdf(x, loc, scale)
joint_b0b1_prior = lambda x0, x1, loc0, loc1, scale0, scale1: b0_prior(x0, loc0, scale0) * b1_prior(x1, loc1, scale1) 


b0_set = np.arange(0.1, 1, 0.005)
b0_mu = 1
b0_sigma = 1

b1_set = np.arange(-0.3, 0.5, 0.005)
b1_mu = 1
b1_sigma = 1


floor = lambda x: 10e-14 if x < 10e-14 else x

NNORMPOSTERIOR = []

for b0_param in b0_set:
 
    for b1_param in b1_set:
        
        print('b0 ', b0_param, 'b1 ', b1_param)
        
        L = [likelihood(y[i], b0_param + b1_param * x[i] , b0_sigma) for i in range(len(y))]
        L = list(map(lambda x: floor(x), L))
        L = np.log(L)
        logLikelihood = np.sum(L)
        
#         b0_pprob = b0_prior(b0_param, b0_mu, b0_sigma)
#         b0_pprob = floor(b0_pprob)
#         logb0_pprob = np.log(b0_pprob)
#         
#         b1_pprob = b1_prior(b1_param, b1_mu, b1_sigma)
#         b1_pprob = floor(b1_pprob)
#         logb1_pprob = np.log(b1_pprob)
        
        joint = joint_b0b1_prior(x0 = b0_param, x1 = b1_param, \
                                 loc0 = b0_mu, loc1 = b1_mu, \
                                 scale0 = b0_sigma, scale1 = b1_sigma)
        
        logjoint = np.log(floor(joint))
        
        lognnormposterior = logLikelihood + logjoint#+ logb0_pprob + logb1_pprob
        
        nnormposterior = np.exp(lognnormposterior)
        
        NNORMPOSTERIOR.append((b0_param, b1_param, nnormposterior))
    
    
df = pd.DataFrame(NNORMPOSTERIOR, columns = ['b0', 'b1', 'nonnorm_jointposterior'])

evidence = df['nonnorm_jointposterior'].sum()

df['jointposterior'] = df['nonnorm_jointposterior'] / evidence

# integrating b1
df['marginalb0'] = df['jointposterior'] * df['b1'].apply(lambda x: b1_prior(x = x, loc = b1_mu, scale = b1_sigma))
dfb0 = df[['b0', 'marginalb0']].groupby(by = ['b0']).sum()
dfb0['marginalb0'] = dfb0['marginalb0'] / dfb0['marginalb0'].sum()
dfb0['priorb0'] = dfb0.index.map(lambda x: b0_prior(x = x, loc = b0_mu, scale = b0_sigma)) 
dfb0['priorb0'] = dfb0['priorb0'] / dfb0['priorb0'].sum()

del df['marginalb0']

# integrating b0 
df['marginalb1'] = df['jointposterior'] * df['b0'].apply(lambda x: b0_prior(x = x, loc = b0_mu, scale = b0_sigma))
dfb1 = df[['b1', 'marginalb1']].groupby(by = ['b1']).sum()
dfb1['marginalb1'] = dfb1['marginalb1'] / dfb1['marginalb1'].sum()
dfb1['priorb1'] = dfb1.index.map(lambda x: b1_prior(x = x, loc = b1_mu, scale = b1_sigma)) 
dfb1['priorb1'] = dfb1['priorb1'] / dfb1['priorb1'].sum()

del df['marginalb1']

# dfb0.plot()
# dfb1.plot()


import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d

x = df['b0'].unique()
y = df['b1'].unique()

X,Y = np.meshgrid(x,y)
Z = np.array(df['jointposterior']).reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z)

ax.set_xlabel('b0')
# ax.set_xlim(-0.52, 1.25)
ax.set_ylabel('b1')
# ax.set_ylim(-0.5,0.75)
ax.set_zlabel('posterior')
ax.set_facecolor((0,0,0,0))
# ax.set_zlim()

plt.show()


print("")

    
    