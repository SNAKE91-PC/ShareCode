'''
Created on Feb 19, 2019

@author: snake91
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as spo
import scipy.special as spe
from functools import partial

import pylab

def MH(func, cntsample): #numsample

    accepted_trials = []
    scale = 5
    
    floor = lambda x: 10e-14 if x < 10e-14 else x
    pdfproposal = lambda x, loc, scale: st.norm.cdf(x, loc = loc, scale = scale)
    qproposal = lambda loc, scale: np.random.normal(loc = loc, scale = scale)
    
    burnin = 100
    counter = 0
#     h = 0.001
    
    #for i in range(numsample):
    i = -1
    while True:
        
        i += 1   
        
        if i == 0:
            
            param_old = qproposal(loc = 0, scale = 1) 
    
            
        param_new = qproposal(loc = param_old, scale = scale)
        
        alpha = min(1, func(param_new) / floor(func(param_old)) * \
                    pdfproposal(param_old, param_new, scale) / floor(pdfproposal(param_new, param_old, scale)))
    
        
        u = np.random.uniform()        
        
        if alpha > u:
    
            if cntsample > burnin:
                accepted_trials.append(param_new)

                            
            param_old = param_new
        
            counter += 1
            
            print(counter, i, sep = ' ')
            if counter == cntsample:
                break
            
    return accepted_trials
    
    
def ecdf(series):
    
    ecdf = []
    series = np.sort(series)

    for i in range(len(series)):
        
        p = i / len(series)
        
        ecdf.append((series[i], p))
    
    return ecdf


def mse(param, x, y):
    
    a,b = param
    
#     x = list(filter(lambda x: x[1] <= q, x))
#     y = list(filter(lambda x: x[1] <= q, y))
    
    x = np.array(list(map(lambda x: x[0], x)))
    y = np.array(list(map(lambda x: x[0], y)))
    
    err = np.mean((a*x - b - y)**2)
    
    print(err)
    return err


    
n = 100000 # distribution of the maximum for a sample size of ten observation normally distributed

samplesize = 20000
f = lambda x: st.norm.cdf(x)**n
g = lambda x: (f(x + 0.00001) - f(x))/ 0.00001
# g = lambda x: n * st.norm.cdf(x)**(n-1) * st.norm.pdf(x)
gumbelpdf = lambda x: np.exp(-x - np.exp(-x))
gumbelquantile = lambda u: -np.log(-np.log(u))
u = np.random.uniform(size = samplesize)


accepted_trials_ordstat = MH(g, samplesize)
ecdf_ordstat = ecdf(accepted_trials_ordstat)

accepted_trials_gumbel = MH(gumbelpdf, samplesize)
ecdf_gumbel = ecdf(accepted_trials_gumbel)

true_gumbel = gumbelquantile(u)
ecdf_truegumbel = ecdf(true_gumbel)

# bounds = ((1/samplesize, 1-1/samplesize), (None, None), (None, None))
param= spo.minimize(fun = mse, x0 = ((0, 0)), args = (ecdf_ordstat, ecdf_truegumbel))#, bounds = bounds)
a,b = param.x
accepted_trials_ordstat1 = a * np.array(accepted_trials_ordstat) - b

print("a", 1/a, "b", b/a, sep = ' ') #"q", q, 

b_n = st.norm.ppf(1-1/n)
a_n = 1/(n * st.norm.pdf(b_n))

print("a_n", a_n, "b_n", b_n, sep = ' ')
accepted_trials_ordstat2 = a_n * np.array(accepted_trials_ordstat) - b_n  

################ TODO: check this
accepted_trials_ordstat3 = (np.percentile(accepted_trials_ordstat,75) - np.percentile(accepted_trials_ordstat,25)) * np.array(accepted_trials_ordstat) - np.median(accepted_trials_ordstat)  
#############################


ecdf_ordstat1 = [(a * i[0] - b, i[1]) for i in ecdf_ordstat]
ecdf_ordstat2 = [((i[0] - b_n) / a_n, i[1]) for i in ecdf_ordstat]
ecdf_ordstat3 = ecdf(accepted_trials_ordstat3)
 
plt.plot(list(map(lambda x: x[0], ecdf_ordstat1)), list(map(lambda x: x[1], ecdf_ordstat1)), label = 'ordstat max N(0,1) n=' + str(n) + " (MH) (emp approx)")
plt.plot(list(map(lambda x: x[0], ecdf_ordstat2)), list(map(lambda x: x[1], ecdf_ordstat2)), label = 'ordstat max N(0,1) n= ' + str(n) + " (MH) (known approx)")


plt.plot(list(map(lambda x: x[0], ecdf_ordstat3)), list(map(lambda x: x[1], ecdf_ordstat3)), label = 'ordstat max N(0,1) n= ' + str(n) + " (MH) (quartile approx)")


# plt.plot(list(map(lambda x: x[0], ecdf_gumbel)), list(map(lambda x: x[1], ecdf_truegumbel)), label = 'gumbel EV (MH)')
plt.plot(list(map(lambda x: x[0], ecdf_truegumbel)), list(map(lambda x: x[1], ecdf_truegumbel)), label = 'gumbel EV ')
plt.legend()

plt.figure()
plt.hist(accepted_trials_ordstat1, bins = 200, histtype='step', label = 'ordstat max N(0,1) n=' + str(n) + " (MH) (emp approx)", normed = True)
plt.hist(accepted_trials_ordstat2, bins = 200, histtype='step', label = 'ordstat max N(0,1) n= ' + str(n) + " (MH) (known approx)", normed = True)
plt.hist(np.array(accepted_trials_gumbel), bins = 200, histtype='step', label = 'gumbel EV (MH)', normed = True)
plt.hist(true_gumbel, bins = 200, histtype= 'step', label = 'gumbel EV (quantile)', normed = True)

plt.legend()





