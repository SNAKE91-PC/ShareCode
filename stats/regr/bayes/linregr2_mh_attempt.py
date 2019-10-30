'''
Created on Aug 31, 2019

@author: snake91
'''


#### likelihood variance assume as theoretical
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

import pandas as pd


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
b0_mu = 10
b0_sigma = 2

b1_prior = lambda x, loc, scale: st.norm.pdf(x, loc, scale)
b1_mu = 5
b1_sigma = 2

sigma_prior = lambda x, loc, scale: st.halfcauchy.pdf(x, loc, scale)
sigma_mu = 0
sigma_sigma = 1

floor = lambda x: 1e-320 if x < 1e-320 else x

np.random.seed(1)


numsample = 10000

cntsample = 0
# b0_cntsample = 0
# b1_cntsample = 0
# sigma_cntsample = 0

accepted_trials = []
# b0_accepted_trials = []
# b1_accepted_trials = []
# sigma_accepted_trials = []

flag_last_accepted = False
# b0_flag_last_accepted = False
# b1_flag_last_accepted = False
# sigma_flag_last_accepted = False

i = 0
param_old = [b0_mu,b1_mu,b0_sigma]
sigmavar = [b0_sigma, b1_sigma, sigma_sigma]
corrmatrix = np.identity(len(sigmavar))

param_new = [0,0,0]


def GaussionCopola(muvector, sigmavar, corrmatrix):
    
    n = len(muvector)
    
    vec = np.random.normal(size = n)
        
    chol = np.linalg.cholesky(corrmatrix)
        
    vec = np.asarray(np.matrix((vec)) * chol)[0]  
    
    vec += muvector
        
#     diag = np.asarray(corrmatrix.diagonal())[0]
    
    newvec = []
    
    for i in range(len(vec)):
        newvec.append( st.norm.cdf(vec[i], loc = muvector[i], scale = sigmavar[i]) )
    
    return newvec


while True:
    
    i+= 1

#     diag = np.asarray(scale_proposal.diagonal())[0]
    param_new = GaussionCopola(muvector = param_old, sigmavar = sigmavar, corrmatrix = corrmatrix)
    param_new[0] = st.norm.ppf(param_new[0], loc = param_old[0], scale = sigmavar[0]) 
    param_new[1] = st.norm.ppf(param_new[1], loc = param_old[1], scale = sigmavar[1])
    param_new[2] = st.halfcauchy.ppf(param_new[2], loc = param_old[2], scale = sigmavar[2])
    
#     param_new[0] = np.random.normal(loc = param_old[0], scale = scale_proposal) #b0_qproposal(b0_old, scale_proposal)
#     param_new[1] = np.random.normal(loc = param_old[1], scale = scale_proposal)
#     param_new[2] = st.halfcauchy.rvs(loc = param_old[2], scale = scale_proposal)
    
#     b0num_old = []
#     b0num_new = []
#     
#     b1num_old = []
#     b1num_new = []
#     
#     sigmanum_old = []
#     sigmanum_new = []
    
    paramnum_old = []
    paramnum_new = []
    for idx in range(len(x)):
        
        res_old = np.log(floor(likelihood(y[idx], param_old[0] + param_old[1] * x[idx], param_old[2])))
        paramnum_old.append(res_old)
        
        res_new = np.log(floor(likelihood(y[idx], param_new[0] + param_new[1] * x[idx], param_new[2])))
        paramnum_new.append(res_new)
        
#         b0num_old.append( resb0_old  )
#         resb0_new = np.log( floor(likelihood(y[idx], b0_new + b1_old * x[idx], sigma_old)))  
#         b0num_new.append( resb0_new )
# 
#         resb1_old = old
#         b1num_old.append( resb1_old ) 
#         resb1_new = np.log(floor(likelihood(y[idx], b0_old + b1_new * x[idx], sigma_old)))
#         b1num_new.append( resb1_new )
#     
#         ressigma_old = old
#         sigmanum_old.append( ressigma_old )
#         ressigma_new = np.log(floor(likelihood(y[idx], b0_old + b1_old * x[idx], sigma_new)))
#         sigmanum_new.append( ressigma_new )
        
    num_div = np.sum(paramnum_new) - np.sum(paramnum_old) +\
                np.log(floor(b0_prior(param_new[0], b0_mu, b0_sigma))) - np.log(floor(b0_prior(param_old[0], b0_mu, b0_sigma))) +\
                np.log(floor(b1_prior(param_new[1], b1_mu, b1_sigma))) - np.log(floor(b1_prior(param_old[1], b1_mu, b1_sigma))) +\
                np.log(floor(sigma_prior(param_new[2], sigma_mu, sigma_sigma))) - np.log(floor(sigma_prior(param_old[2], sigma_mu, sigma_sigma)))
#     b0num_div = np.sum(b0num_new) - np.sum(b0num_old) + np.log(floor(b0_prior(b0_new, b0_mu, b0_sigma))) - np.log(floor(b0_prior(b0_old, b0_mu, b0_sigma)))
#     b1num_div = np.sum(b1num_new) - np.sum(b1num_old) + np.log(floor(b1_prior(b1_new, b1_mu, b1_sigma))) - np.log(floor(b1_prior(b1_old, b1_mu, b1_sigma)))
#     sigmanum_div = np.sum(sigmanum_new) - np.sum(sigmanum_old) + np.log(floor(sigma_prior(sigma_new, sigma_mu, sigma_sigma))) - np.log(floor(sigma_prior(sigma_old, sigma_mu, sigma_sigma)))
    
    acceptance_ratio = cntsample / i
#     b0_acceptance_ratio = b0_cntsample / i
#     b1_acceptance_ratio = b1_cntsample / i
#     sigma_acceptance_ratio = sigma_cntsample / i
    
#     def calcScaleProposal(acceptance_ratio, scale_proposal, flag_last_accepted = False, startignored = 50):
#         
#         if i > startignored:
#             if acceptance_ratio < 0.1 and flag_last_accepted == True:
#                 scale_proposal *= 0.2
#             elif acceptance_ratio >= 0.1 and acceptance_ratio < 0.1 and flag_last_accepted == True:
#                 scale_proposal *= 0.5
#             elif acceptance_ratio >= 0.2 and acceptance_ratio < 0.2 and flag_last_accepted == True:
#                 scale_proposal *= 0.9
#             elif acceptance_ratio > 0.3 and flag_last_accepted == True:
#                 scale_proposal *= 1.2
#             
#             
#         return scale_proposal


    res = num_div + \
                (np.log(floor(st.norm.pdf(param_old[0], param_new[0], sigmavar[0]))) - np.log(floor(st.norm.pdf(param_new[0], param_old[0], sigmavar[0]) ))) + \
                (np.log(floor(st.norm.pdf(param_old[1], param_new[1], sigmavar[1]))) - np.log(floor(st.norm.pdf(param_new[1], param_old[1], sigmavar[1])))) + \
                (np.log(floor(st.halfcauchy.pdf(param_old[2], param_new[2], sigmavar[2]))) - np.log(floor(st.halfcauchy.pdf(param_new[2], param_old[2], sigmavar[2]))))
                
                
#     b0_res = b0num_div + \
#                 (np.log(floor(st.norm.pdf(b0_old, b0_new, b0_scale_proposal))) - np.log(floor(st.norm.pdf(b0_new, b0_old, b0_scale_proposal) )))
#                 
#     b1_res = b1num_div + \
#                 (np.log(floor(st.norm.pdf(b1_old, b1_new, b1_scale_proposal))) - np.log(floor(st.norm.pdf(b1_new, b1_old, b1_scale_proposal))))
#                 
#     sigma_res = sigmanum_div + \
#                 (np.log(floor(st.halfcauchy.pdf(sigma_old, sigma_new, sigma_scale_proposal))) - np.log(floor(st.halfcauchy.pdf(sigma_new, sigma_old, sigma_scale_proposal))))
                
    alpha = min(0, res)
#     b0_alpha = min(0, b0_res)
#     b1_alpha = min(0, b1_res)
#     sigma_alpha = min(0, sigma_res)

#     scale_proposal = calcScaleProposal(acceptance_ratio, scale_proposal, flag_last_accepted)
#     b0_scale_proposal = calcScaleProposal(b0_acceptance_ratio, b0_scale_proposal, b0_flag_last_accepted)
#     b1_scale_proposal = calcScaleProposal(b1_acceptance_ratio, b1_scale_proposal, b1_flag_last_accepted)
#     sigma_scale_proposal = calcScaleProposal(sigma_acceptance_ratio, sigma_scale_proposal, sigma_flag_last_accepted)
    
    u = np.random.uniform()        
    
    b0str    = "b0    " + "old " + str(param_old[0])     + " new " + str(param_new[0])    +" "+ str(cntsample)    +" "+ str(i)  #+ " scale " + str(scale_proposal)
    b1str    = "b1    " + "old "  + str(param_old[1])    + " new " + str(param_new[1])    +" "+ str(cntsample)    +" "+ str(i) #+ " scale " + str(scale_proposal)
    sigmastr = "sigma " + "old "  + str(param_old[2]) + " new " + str(param_new[2]) +" "+ str(cntsample) +" "+ str(i) #+ " scale " + str(scale_proposal)
    
#     maxlen = max([len(b0str), len(b1str), len(sigmastr)])
    
#     b0str = b0str.ljust(maxlen)
#     b1str = b1str.ljust(maxlen)
#     sigmastr = sigmastr.ljust(maxlen)
    
    print(b0str)
    print(b1str)
    print(sigmastr)
    
    flag_last_accepted = False
    print(" ")
#     b0_flag_last_accepted = False
#     b1_flag_last_accepted = False
#     sigma_flag_last_accepted = False

    if alpha > np.log(u):
        cntsample += 1
        
        accepted_trials.append(tuple(param_new))#, np.exp(b0num_new)))
        
        param_old = param_new
    
        flag_last_accepted = True
    
        if len(accepted_trials) > 3:
            corrmatrix = np.matrix(pd.DataFrame(accepted_trials).corr())
            sigmavar = np.array(pd.DataFrame(accepted_trials).apply(lambda x: np.sqrt(np.var(x))))
#     if b0_alpha > np.log(u):
# 
#         b0_cntsample += 1
#         
#         b0_accepted_trials.append(b0_new)#, np.exp(b0num_new)))
#         
#         b0_old = b0_new
#     
#         b0_flag_last_accepted = True
#     if b1_alpha > np.log(u):
# 
#         b1_cntsample += 1
#         
#         b1_accepted_trials.append(b1_new)#, np.exp(b0num_new)))
#         
#         b1_old = b1_new
#     
#         b1_flag_last_accepted = True
#         
#     if sigma_alpha > np.log(u):
# 
#         sigma_cntsample += 1
#         
#         sigma_accepted_trials.append(sigma_new)#, np.exp(b0num_new)))
#         
#         sigma_old = sigma_new
#     
#         sigma_flag_last_accepted = True
# 
#     if b0_cntsample >= numsample and b1_cntsample >= numsample and sigma_cntsample >= numsample:
#         break




b0_data = pd.DataFrame(accepted_trials, columns = ['b0'])
b0_data.to_csv("/home/snake91/data_w_assumedvariance.csv")

b1_data = pd.DataFrame(accepted_trials, columns = ['b1'])
b1_data.to_csv("/home/snake91/data_w_assumedvariance.csv")

sigma_data = pd.DataFrame(accepted_trials, columns = ['sigma'])
sigma_data.to_csv("/home/snake91/data_w_assumedvariance.csv")


print("")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    