'''
Created on Dec 9, 2018

@author: snake91
'''

#this is pippo 
 
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf, ccf
import itertools
import scipy.optimize as opt
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np

from mle import likelihood as logL
from mle import constraint as cons

data = pd.read_csv(r'/home/snake91/git/ShareCode/autocorr/DJI.csv')
data['Date'] = pd.to_datetime(data['Date'])

scenario = {}
scenario['Lehman'] = ('2007-04-01', '2008-12-31')

data = data[(data['Date'] > scenario['Lehman'][0]) & (data['Date'] < scenario['Lehman'][1] )]

data['Return'] = data['Close'] / data['Close'].shift() - 1
data.dropna(inplace = True)

# plot_pacf(data['Return'].get_values(), lags = 10)
# plot_acf(data['Return'].get_values(), lags = 10)


class Helper(object):
    
    def __init__(self, window):
        
        self.result = [np.nan for i in range(window-1)] #@Unusedvariable
#         self.status = [np.nan for i in range(window-1)] #@Unusedvariable
        self.count = 0
        
    def append(self, obj):
        
        print(self.count, obj)#.x, obj.message, sep = ' ')
        self.result.append(obj)#.x)
#         self.status.append(obj.message)
        
        self.count += 1
        
        
        
        return 1
        
    
def optimize(x, process, maxLag = None, args = None):

#     x = np.asmatrix(x)
#     
#     nprocess = x.shape[0]
#     pLag = args[0]
#     qLag = args[1]
#     
#     assert(maxLag is None or args is None)
#     
#     if maxLag is None:
#         maxLag = 5
#         # compute cross-correlation matrix
# 
#         comb = list(itertools.product(x, repeat = x.shape[0]))
#         size = int(np.sqrt(len(comb)))
# 
#         u = np.zeros(shape = (size**2, 1, maxLag))
# 
#         if len(comb) == 1:
#             u[0] = acf( comb[0][0] )[:maxLag]
#         else:
#             for i in range(len(comb)):
#                 u[i] = ccf( comb[i][0], comb[i][1] )[:maxLag]  
#     
#         # attempt auto-identification
#         
#         
#         u = np.reshape(u, newshape = (size, size, maxLag))    # cross-corr matrix
# 
#         # model auto-identification to be implemented
# 
#     x0 = [0. for i in range(sum(args))]  #@Unusedvariable
#       
#     if process.__name__.endswith('N'): # TODO: GENERALIZE VARMA WITH ANY DISTRIBUTION
#         for i in range(nprocess): #@Unusedvariable
#             x0.append( np.sqrt(np.var(x)) )
#         
#     x0 = tuple(x0)
#     
#     bounds = [ (-0.9, 0.9) for i in range(sum(args)) ]# + [(0.0001, None)] #@Unusedvariable 
#     
#     if process.__name__.endswith('N'):
#         for i in range(nprocess): #@Unusedvariable
#             bounds += [(0.000001, None)]
#              
# #     if process.func_name.endswith('T'):
# #         bounds += [(0.01, None)] #df
#         
#     bounds = tuple(bounds)
#     
#     #constraints = ({'type': 'ineq', 'fun': lambda y: 1 - np.sum(np.abs(y))})
#         
#     args = tuple([x] + list(args))
#     obj = opt.minimize(process, x0 = x0, 
#                        args = args, 
#                        bounds = bounds, 
#                        constraints = ({'type': 'ineq', 
#                                             'fun': lambda params: cons.consVARp(params[: nprocess**2 * pLag], pLag)},
#                                        {'type': 'ineq', 
#                                             'fun': lambda params: cons.consVMAq(params[nprocess**2 * pLag: nprocess**2 * (pLag + qLag)], qLag)}
#                                     ))#, method = "L-BFGS-B")
    x = x.reshape(1,250)
    obj = process(x, p = args[0], q = args[1])
    
    return obj
    
    
# print '\nstarting ARpN...\n'
# dummy = Helper(window = 250)
# data['ARpN'] = data['Return'].rolling(window = 250).apply(lambda x: dummy.append(optimize(x, logL.maxARpN)))
#     
#     
# data['ARpN'] = dummy.result
# data['statusARpN'] = dummy.status
#   
# print '\nstarting MAqN...\n'
# dummy = Helper(window = 250)
# data['MAqN'] = data['Return'].rolling(window = 250).apply(lambda x: dummy.append(optimize(x, logL.maxMAqN)))
#   
# data['MAqN'] = dummy.result

print('\nstarting ARMApqN...\n')

dummy  = Helper(window = 250)
data['ARMApqN'] = data['Return'].rolling(window = 250).apply(lambda x: dummy.append(optimize(x, logL.maxVARMApqN, args = (2,2)))) # VARMA now works also for 1-dim processes
data['ARMApqN'] = dummy.result
 

# print('\nstarting ARMApqT...\n')
# 
# dummy = Helper(window = 250)
# data['ARMApqT'] = data['Return'].rolling(window = 250).apply(lambda x: dummy.append(optimize(x, logL.maxARMApqT, args = (2,2))))
# data['ARMApqT'] = dummy.result



print(data)



















