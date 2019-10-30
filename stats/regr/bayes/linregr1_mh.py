'''
Created on Aug 31, 2019

@author: snake91
'''


#### likelihood variance assume as theoretical
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

b0 = 5
b1 = 0.1
size = 1000
scale = 10
eps = np.random.normal(size = size, scale = scale)
x = np.arange(0, len(eps))

y = b0 + b1 * x + eps

#### bayes

likelihood = lambda x, loc, scale: st.norm.pdf(x, loc, scale)
b0_prior = lambda x, loc, scale: st.norm.pdf(x, loc, scale)

# b0_set = np.arange(0, 2, 0.01)
b0_mu = 10
b0_sigma = 2

floor = lambda x: 1e-320 if x < 1e-320 else x

np.random.seed(1)

b0_pdfproposal = lambda x, loc, scale: st.norm.pdf(x, loc = loc, scale = scale)
# b0_qproposal = lambda loc, scale: 

# posterior = lambda obs, obsloc, obsscale, param, muparam, scaleparam: likelihood(obs, obsloc, obsscale) * b0_prior(param, muparam, scaleparam)
# totnumsample = 100000
numsample = 10000

scale_proposal = 1
cntsample = 0
accepted_trials = []
# burnin = 100
# for i in range(1,totnumsample):
i = 0
flag_last_accepted = False

while True:
    
    i+= 1
    
    if i == 1:
        scale_proposal = b0_sigma
        b0_old = np.random.normal(loc = b0_mu, scale = b0_sigma)#b0_qproposal(np.mean(y), scale_proposal)#b0_init
    
    b0_new = np.random.normal(loc = b0_old, scale = scale_proposal) #b0_qproposal(b0_old, scale_proposal)
    
    b0num_old = []
    b0num_new = []
#     b0_new = 20
    for idx in range(len(x)):
#     b0num_old = np.sum([np.log(likelihood(y[i], b0_old + b1 * x[i], scale)) + np.log(b0_prior(b0_old, b0_mu, b0_sigma)) for i in range(len(x))])
#     b0num_new = np.sum([np.log(likelihood(y[i], b0_new + b1 * x[i], scale)) + np.log(b0_prior(b0_new, b0_mu, b0_sigma)) for i in range(len(x))])
        resb0_old = np.log(floor(likelihood(y[idx], b0_old + b1 * x[idx], scale))) 
#         if np.isinf(resb0_old):
#             break
        b0num_old.append( resb0_old  )
    
        resb0_new = np.log( floor(likelihood(y[idx], b0_new + b1 * x[idx], scale)))  
#         if np.isinf(resb0_new):
#             break
        b0num_new.append( resb0_new )
    
#     if (idx + 1) < len(x):
#         mean_prior = 0.5
#         scale_prior = 1
#         scale_proposal = scale_prior
#         b0_old = np.random.normal(loc = mean_prior, scale = scale_prior)#b0_qproposal(np.mean(y), scale_proposal)#b0_init
    
    b0num_div = np.sum(b0num_new) - np.sum(b0num_old) + np.log(floor(b0_prior(b0_new, b0_mu, b0_sigma))) - np.log(floor(b0_prior(b0_old, b0_mu, b0_sigma)))
    
#     b0num_div = np.prod([
#                             posterior(y[i], b0_new + b1 * x[i], scale, b0_new, b0_mu, b0_sigma) / 
#                             posterior(y[i], b0_old + b1 * x[i], scale, b0_old, b0_mu, b0_sigma)
#                                 for i in range(len(x))
#                         ])
#     
    
    
#     b0num_div = np.prod([
#                             (likelihood(y[i], b0_new + b1 * x[i], scale) * b0_prior(b0_new , b0_mu, b0_sigma)) / 
#                             floor(likelihood(y[i], b0_old + b1 * x[i], scale) * b0_prior(b0_old , b0_mu, b0_sigma))
#                                    
#                               for i in range(len(x))
#                           ]) 
    
    
#     b0num_new = np.prod([likelihood(y[i], b0_old + b1 * x[i], scale) *\
#                              b0_prior(b0_old , b0_mu, b0_sigma)  for i in range(len(x))])
#     
#     b0num_old = np.prod([likelihood(y[i], b0_old + b1 * x[i], scale) *\
#                              b0_prior(b0_old , b0_mu, b0_sigma)  for i in range(len(x))])
    
#     b0num_div = b0num_new / b0_old
# b0num_new / floor(b0num_old) * \
 
#     print(b0num_div)

    acceptance_ratio = cntsample / i

    res = b0num_div + \
                (np.log(floor(b0_pdfproposal(b0_old, b0_new, scale_proposal))) - np.log(floor(b0_pdfproposal(b0_new, b0_old, scale_proposal) )))
    alpha = min(0, res)

    u = np.random.uniform()        
    
    print("old", b0_old, "new", b0_new, cntsample, i, "scale ", scale_proposal,  sep = ' ') #b0_new, b0num_div, 
    
    
    if i > 50:   
        if acceptance_ratio < 0.05 and flag_last_accepted == True:
            scale_proposal *= 0.2
        elif acceptance_ratio >= 0.05 and acceptance_ratio < 0.1 and flag_last_accepted == True:
            scale_proposal *= 0.5
        elif acceptance_ratio >= 0.1 and acceptance_ratio < 0.2 and flag_last_accepted == True:
            scale_proposal *= 0.9
        elif acceptance_ratio > 0.3 and flag_last_accepted == True:
            scale_proposal *= 1.2
            
            
    flag_last_accepted = False
    if alpha > np.log(u):

#         if cntsample > burnin:
        cntsample += 1
        
        accepted_trials.append(b0_new)#, np.exp(b0num_new)))
        
        b0_old = b0_new
    
        flag_last_accepted = True
        if cntsample == numsample:
            break
            

import pandas as pd

data = pd.DataFrame(accepted_trials, columns = ['b0'])
data.to_csv("/home/snake91/data_w_assumedvariance.csv")

print("")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    