'''
Created on Oct 11, 2019

@author: snake91
'''


import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


  
def em_algorithm(X, states, pdf, maxiter = 2500):
    
    
    k = states  
    weights = np.ones((k)) / k
    means = np.random.choice(X, k)
    variances = np.random.random_sample(size=k)
    eps=1e-8
    for step in range(maxiter): # @Unusedvariable
        # calculate the maximum likelihood of each observation xi
        likelihood = []
      
        print(means, variances, weights, sep = " ")
        # Expectation step
        for j in range(k):
            likelihood.append(pdf(X, means[j], np.sqrt(variances[j])))
        likelihood = np.array(likelihood)
        
        b = []
        # Maximization step 
        for j in range(k):
            # use the current values for the parameters to evaluate the posterior
            # probabilities of the data to have been generanted by each gaussian    
            b.append((likelihood[j] * weights[j]) / (np.sum([likelihood[i] * weights[i] for i in range(k)], axis=0)+eps))
          
            # updage mean and variance
            means[j] = np.sum(b[j] * X) / (np.sum(b[j]+eps))
            variances[j] = np.sum(b[j] * np.square(X - means[j])) / (np.sum(b[j]+eps))
            
            # update the weights
            weights[j] = np.mean(b[j])
                
                
    return weights, means, variances



n_samples = 1000000
mu1, sigma1 = -4, 1.2 # mean and variance
mu2, sigma2 = 4, 1.8 # mean and variance
mu3, sigma3 = 0, 1.6 # mean and variance

x1 = np.random.normal(mu1, np.sqrt(sigma1), n_samples)
x2 = np.random.normal(mu2, np.sqrt(sigma2), n_samples)
x3 = np.random.normal(mu3, np.sqrt(sigma3), n_samples)

X = np.array(list(x1) + list(x2) + list(x3))
np.random.shuffle(X)
        
def pdf(data, mean: float, variance: float):
    s1 = 1/(np.sqrt(2*np.pi*variance))
    s2 = np.exp(-(np.square(data - mean)/(2*variance)))
    return s1 * s2
  
# print(means, variances)
  
k = 3

weights, means, variances = em_algorithm(X = X, states = k, pdf = pdf, maxiter = 2500)
    
plt.hist(X, bins = 2000)
print("")














