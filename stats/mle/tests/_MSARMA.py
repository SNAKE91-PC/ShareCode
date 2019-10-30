# '''
# Created on Feb 27, 2019
# 
# @author: snake91
# '''
# 
# from mle import simulate as sim
# 
# import numpy as np
# import matplotlib.pyplot as plt
# import functools
# import pandas as pd
# import scipy.optimize as opt
# 
# import mle.likelihood as logL
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa import ar_model
# 
# 
# def pdf(x, mean: float, variance: float):
#     s1 = 1/(np.sqrt(2*np.pi*variance))
#     s2 = np.exp(-(np.square(x - mean)/(2*variance)))
#     return s1 * s2
# 
# 
# def em_algorithm(X, states, pdf, maxiter = 2500):
#     
#     # currently only AR1N
#     X = np.array(X)
#     
#     k = states
#     weights = np.ones((k)) / k
#     
#     eps=1e-8
#     
#     uncsigma = np.sqrt(np.var(X))
#     sigmas = [uncsigma] * k
#     params = [0] * k
#     
#     for step in range(maxiter): # @Unusedvariable
#         # calculate the maximum likelihood of each observation xi
#         likelihood = []
#       
# #         if np.min(sigmas) <= 0:
# #             sigmas[sigmas.index(np.min(sigmas))] += 1e-4
# #             continue
#         
#         print(params, sigmas, weights, sep = " ")
#         # Expectation step
#         for j in range(k):
#             # to be generalized for ARp
#             likelihood.append(logL.maxARpN([params[j]], X, sigma = sigmas[j], estimation='EM') )#pdf(X, means[j], np.sqrt(variances[j])))
#         likelihood = np.exp(np.array(likelihood))
#         
#         b = []
#         # Maximization step 
#         for j in range(k):
#             # use the current values for the parameters to evaluate the posterior
#             # probabilities of the data to have been generanted by each gaussian    
#             b.append((likelihood[j] * weights[j]) / (np.sum([likelihood[i] * weights[i] for i in range(k)], axis=0)+eps))
#           
#             # updage mean and variance
#             # subtract the prediction
#             params[j] = np.sum(b[j] * X[1:]) / (np.sum(b[j]+eps)) # - prediction
#             
#             
#             sigmas[j] = np.sum(b[j] * np.square(X[1:] - params[j])) / (np.sum(b[j]+eps)) # - params[j] - prediction*
#             
#             # update the weights
#             weights[j] = np.mean(b[j])
#                 
#                 
#     return weights, params, sigmas
# 
# 
# 
# 
# 
# def maxMSAR11(params, x):
#     
# #     a0 =  0.9
# #     a1 =  -0.9
# #     p00 = 0.9
# #     p01 = 0.1
# #     p10 = 0.1
# #     p11 = 0.9
#     
#     
#     a0 = params[0]
#     a1 = params[1]
#  
#     condMatrix = params[2:]
#     p00 = condMatrix[0]
#     p01 = condMatrix[1]
#     p10 = condMatrix[2]
#     p11 = condMatrix[3]
#     
#     
#     L = []    
#     variance = np.var(x) 
#     for t in range(len(x)):
#         a = np.log(pdf(x[t], a0 * x[t-1], variance)) + np.log(p00)  # probability of going from 0 to 0 # 0-process # p00
#         b = np.log(pdf(x[t], a1 * x[t-1], variance)) + np.log(p01)  # probability of going from 0 to 1 # 1-process # p01
#         c = np.log(pdf(x[t], a0 * x[t-1], variance)) + np.log(p10)  # probability of going from 1 to 0 # 0-process # p10
#         d = np.log(pdf(x[t], a1 * x[t-1], variance)) + np.log(p11)  # probability of going from 1 to 1 # 1-process # p11
#         
#         res = a + b + c + d
#     
#         L.append(res)
#     
#     L = -np.sum(L)
#     
#     print(a0, a1, p00, p01, p10, p11, L)
#     return L
# 
# 
# 
# 
# 
# 
# 
# 
# 
# condMatrix = { 0 : [0.5, 0.5], 
#                1 : [0.5, 0.5] }
# 
# # condMatrix = {0 : [1]}
# 
# t = 100000
# 
# p1 = [0.9]
# p2 = [-0.9]
# 
# paramsp = [p1,p2]
# 
# np.random.seed(10)
# 
# x = sim.msarmaN(t, condMatrix, 0, paramsp)#, paramsq) #mixed process
# 
# k = len(condMatrix.keys())
# 
# # weights, means, variances = em_algorithm(X = x, states = k, pdf = pdf, maxiter = 2500)
# 
# x0 = [0., 0., 0.5, 0.5, 0.5, 0.5]
# bounds = [(-1, 1), (-1, 1)] + [(0, 1)] * len(condMatrix.keys())**2
# 
# 
# def consprob( x ):
#     
#     prob = x[2:]
#     
#     prob = np.array(prob).reshape(2,2)
#     
# #     print(1 -np.max( np.sum(prob, axis = 0) ))
#     return 1 -np.max( np.sum(prob, axis = 0) )
# 
# params = opt.minimize(fun = maxMSAR11, x0 = x0, args = x, bounds = bounds, constraints = {'type':'ineq', 'fun': lambda x: consprob(x)})
# 
# 
# msiid = sim.msiidN(t = t, pmatrix = condMatrix, startstate = 0, paramsmean = [0,0], paramsvar = [1,4])
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

