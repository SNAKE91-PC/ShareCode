'''
Created on 8 Nov 2019

@author: snake91

python translation of the EM algorithm (R) here
https://thesamuelsoncondition.com/2016/02/20/time-series-v-the-hamilton-filter/
'''


import pandas as pd
import numpy as np
import scipy.optimize as spo
from mle.simulate import msiidN, sim_markovchain

import matplotlib.pyplot as plt

from copy import deepcopy

N = lambda x, mu, sigma : (1./np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2 / (2*sigma**2))


def mrs_est(theta, x, y, option = 'MLE'):
     
    alpha1, alpha2, beta1, beta2, sigma1, sigma2, p11, p22 = theta
    p11 = 1/(1+np.exp(-p11))
    p22 = 1/(1+np.exp(-p22))

    #in order to make inference about what state we are in in period t we need the conditional
    # densities given the information set through t-1
    
    f1 = N(y, alpha1 + beta1 * x, sigma1)
    f2 = N(y, alpha2 + beta2 * x, sigma2)

    f = np.asarray([f1,f2]).T
    
        #S.forecast is the state value looking forward conditional on info up to time t
    #S.inf is the updated state value
      
    S_forecast = np.zeros(shape = (len(y), 2))
    S_inf = deepcopy(S_forecast)
    ov = np.ones(2)
    
    P = np.asarray([[p11, 1-p11], [1-p22, p22]])
    model_lik = np.zeros(len(y))
    
    S_inf[0, ] = (np.diag(P) * f[0,]) / np.dot(ov, np.diag(P) * f[0,].T)
    
    
    for i in range(1, len(y)):

        #in time t we first make our forecast of the state in t+1 based on the
        # data up to time t, then we update that forecast based on the data
        # available in t+1
        
        S_forecast[i,:] = np.dot(P, S_inf[i-1: i, :].T).T
        S_inf[i, : ] = (S_forecast[i,: ] * f[i,:]) / np.dot(S_forecast[i, :], f[i, :])
            
        model_lik[i] = max(1e-400, np.dot(ov, S_forecast[i] * f[i]))
     
    if option == 'MLE':
        logl = np.sum(np.log(model_lik[1:]))
         
        print(theta, -logl, sep = '')
        
        return -logl
    elif option == 'SE':
        
        return (S_inf, S_forecast)


def ham_smooth(theta, x, y):
    
    alpha1, alpha2, beta1, beta2, sigma1, sigma2, p11, p22 = theta
    S_inf, S_forecast = mrs_est(theta, x, y, option = 'SE')
    
    T = len(y)
    P_smooth = np.zeros(2 * len(y)).reshape((len(y), 2)) 
    P_smooth[T-1: ] = deepcopy(S_inf[T-1: ])
     
    for i in reversed(range(0, T-1)):
         
        #1. probability that we observe S(t)=1, S(t+1)=1
        #2. probability that we observe S(t)=1, S(t+1)=2
        #3. probability that we observe S(t)=2, S(t+1)=1
        #4. probability that we observe S(t)=2, S(t+1)=2
         
        #for #1 P[S(t)=1,S(t+1)=1|I(t+1)] = {P[S(t+1)=1|I(t+1)]*P[S(t)=1|I(t)]*P[S(t+1)=1|S(t)=1]}/P[S(t+1)=1|I(t)]
        p1 = (S_inf[i, 0] * S_inf[i-1, 0] * p11)/ S_forecast[i,0]
  
        #for #2 we have
        p2 = (S_inf[i, 1] * S_inf[i-1, 0] * (1-p11)) / S_forecast[i, 1]
         
        #for #3 we have
        p3 = (S_inf[i, 0] * S_inf[i-1, 1]) * (1-p22) / S_forecast[i, 0]
         
        #for #4 we have
        p4 = (S_inf[i, 1] * S_inf[i-1, 1] * p22) / S_forecast[i, 1]
         
        P_smooth[i, 0] = p1 + p2
        P_smooth[i, 1] = p3 + p4
         
     
    return P_smooth
    
    
    
path = r'/home/snake91/ms/'
  
y = pd.read_csv(path + 'lng.csv')
x = pd.read_csv(path + 'oil.csv')
      
y['lng'] = np.log(y['lng'])
x['oil'] = np.log(x['oil'])
      
x = x[x['date'] >= np.min(y['date'])]
  
x = np.array(x['oil'])
y = np.array(y['lng'])
 
theta_correct = [0.0985075, 
         -0.0209378,
         0.4512085,
         0.2893281,
         0.1971233,
         0.1935434,
         0.9775813,
  0.9888376]




print(mrs_est(theta_correct, x, y))
theta1 =  [0.00517815, 0.24865985, 0.27892527, 0.41002538, 0.17788533, 0.19907279, 1.        , 1.        ]
print(mrs_est(theta1, x, y))
# theta = [0.9, 0.9] #5, -5, 0, 0, 1, 1, 

# alpha1, alpha2, beta1, beta2, sigma1, sigma2, p11, p22 = theta  

#### SIMULATE PROCESSES
np.random.seed(10)

transmat = {0: [0.5, 0.5], 1: [0.5, 0.5]} #np.matrix([[0.9, 0.1], [0.2, 0.8]])

x1 = np.random.normal(size = 5000, loc = 0, scale = 1)
x2 = np.random.normal(size = 5000, loc = 0, scale = 1)

# x1 = np.linspace(0, 2, 5000)
# x2 = np.linspace(2, 4, 5000)

transitions = sim_markovchain(t = 5000, pmatrix = transmat, startvalue = 0)

states = {0 : x1, 1: x2}
beta = [1, 4]
intercept = [-5, 10 ]
# x = np.array([states[transitions[i]][i] for i in range(len(transitions))])
# y = np.array([intercept[transitions[i]] + beta[transitions[i]] * states[transitions[i]][i] + np.random.normal() for i in range(len(transitions))])


x = np.hstack([x1,x2])
y1 = [intercept[0] + beta[0] * states[0][i] + np.random.normal() for i in range(len(transitions))]
y2 = [intercept[1] + beta[1] * states[1][i] + np.random.normal() for i in range(len(transitions))]

y = np.hstack([y1, y2])





theta_new = [-5, 10, 1, 4, 1, 1, 0.5, 0.5]
mrs_est(theta_new, x, y) #18941.68118

theta = []
bounds = []

cons = {'type': 'eq',  'fun' :lambda x: cons(x)}
# theta = [0, 0, 0, 0, 1, 1, 0.5, 0.5]  
theta = [-5, 10, 1, 4, 1, 1, 0.5, 0.5]
bounds = [(-10,0), (5,10), (-1, 1), (-1, 5), (1e-8, 2), (1e-8, 2), (0,1), (0,1)]  


# res1 = spo.minimize(mrs_est, x0 = theta, args = (x,y), bounds = bounds, tol = 10e-8)
# print(res1)

res = spo.differential_evolution(func = mrs_est, bounds = bounds, args = (x,y), atol = 10e-16)
psmooth = ham_smooth(theta, x, y)
print(res)

print("")

# res1 = ham_smooth(res.x, x, y)


# alpha1 = 0.0985075 
# alpha2 = -0.0209378  
# beta1 = 0.4512085
# beta2 =  0.2893281
# sigma1 = 0.1971233 
# sigma2 = 0.1935434 
# p11 = 0.9775813 
# p22 = 0.9888376
#     
# theta = np.array([alpha1, alpha2, beta1, beta2, sigma1, sigma2, p11, p22])

# 
# mrs_est(theta, x, y)
# mrs_est(res.x, x, y)


# res2 = ham_smooth(theta, x, y)

# plt.plot(res1[:, 1], label = 'io')    
# plt.plot(res2[:, 1], label = 'modulo R')    
# plt.legend()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    