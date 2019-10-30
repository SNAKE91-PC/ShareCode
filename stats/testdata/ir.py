'''
Created on Feb 21, 2019

@author: snake91
'''

import matplotlib.pyplot as plt
import pandas as pd
import quandl
import numpy as np
import scipy.optimize as spo
# from lmfit import Parameters, Model

key = 'Mvr771sxAVihc7fG2XZE'

# startDate = '2017-12-10'
# endDate   = '2018-01-01'
# 
# 
# symbols = ['BOF/QS_D_IEUEONIA', 
#            'BOF/QS_D_IEUTIO1M',
#            'BOF/QS_D_IEUTIO2M',
#            'BOF/QS_D_IEUTIO3M',
#            'BOF/QS_D_IEUTIO6M',
#            'BOF/QS_D_IEUTIO8M',
#            'BOF/QS_D_IEUTIO9M',
#            'BOF/QS_D_IEUTIO1A',
#            'BOF/QS_D_IEUTIO2A'
#            
#            
#            ]
# 
# data = quandl.get(symbols, authtoken=key, start_date = startDate, end_date = endDate)
# 
# data = pd.DataFrame(data)
# data.dropna(inplace = True)


def NSS(params, t):
    
    b1 = params[0]
    b2 = params[1]
    b3 = params[2]
    b4 = params[3]
    lambda1 = params[4]
    lambda2 = params[5]
    
    y = b1 + \
         b2 *  ((1-np.exp(-t/lambda1)) / (t/lambda1)) + \
         b3 * (((1-np.exp(-t/lambda1)) / (t/lambda1)) - np.exp(-t/lambda1)) + \
         b4 * (((1-np.exp(-t/lambda2)) / (t/lambda2)) - np.exp(-t/lambda2)) 
      
    
    return y

def MSE(params, t, x, func):
    
    y = func(params, t)
    print(params, np.mean((y-x)**2), sep = ' ')
    return np.mean((y - x)**2)



rng = np.random.normal(size = 1000)/1000.

bounds = [(0, 15), (-15, 15), (0, 30), (0, 30), (0, 30), (0, 30)]

params = []

for param in bounds:
    sampleParam = np.random.uniform(size = 1000, low = param[0], high = param[1]).tolist()
    
    params.append(sampleParam)
    
params = list(zip(*params))



print('')










































