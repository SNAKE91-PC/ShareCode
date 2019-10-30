'''
Created on Jun 1, 2019

@author: snake91
'''


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

from mle.mleclass import mleobj

data1 = pd.read_csv(r'/home/snake91/git/ShareCode/autocorr/DJI.csv')
data2 = pd.read_csv(r'/home/snake91/git/ShareCode/autocorr/NASDAQ.csv')

data1['Date'] = pd.to_datetime(data1['Date'])
data2['Date'] = pd.to_datetime(data2['Date'])

scenario = {}
scenario['Lehman'] = ('2007-04-01', '2008-12-31')

data1 = data1[(data1['Date'] > scenario['Lehman'][0]) & (data1['Date'] < scenario['Lehman'][1] )]
data2 = data2[(data2['Date'] > scenario['Lehman'][0]) & (data2['Date'] < scenario['Lehman'][1] )]


data1['Return'] = data1['Close'] / data1['Close'].shift() - 1
data1.dropna(inplace = True)

data2['Return'] = data2['Close'] / data2['Close'].shift() - 1
data2.dropna(inplace = True)

data = pd.merge(data1, data2, how = 'inner', on = 'Date', suffixes=('_DJI', '_NASDAQ'))

data = data[['Date'] + list(filter(lambda x: 'Return' in x, data.columns))]

data.set_index(['Date'], inplace = True)
# data['Return'] = list(np.array(data[list(filter(lambda x: 'Return' in x, data.columns))]))
# data = data[['Date', 'Return']]



class Helper(object):
    
    def __init__(self, window):
        
        self.result = [np.nan for i in range(window-1)] #@Unusedvariable
#         self.status = [np.nan for i in range(window-1)] #@Unusedvariable
        self.count = 0
        
    def append(self, obj):
        
#         print(self.count, obj)#.x, obj.message, sep = ' ')
        self.result.append(obj)#.x)
#         self.status.append(obj.message)
        
        self.count += 1
        
        
        
        return 1
        
    
def rollaccf(x, maxLag = 1):

    obj = mleobj._accfMatrix(process = x, maxLag=maxLag)
    
    return obj

def rollpccf(x, maxLag = 1):
    
    obj = mleobj._pccfMatrix(process = x, maxLag=maxLag)
    
    return obj


from numpy.lib.stride_tricks import as_strided as stride

def roll(df, w, **kwargs):
    v = df.values
    d0, d1 = v.shape
    s0, s1 = v.strides

    a = stride(v, (d0 - (w - 1), w, d1), (s0, s0, s1))

    rolled_df = pd.concat({
        row: pd.DataFrame(values, columns=df.columns)
        for row, values in zip(df.index, a)
    })

    return rolled_df.groupby(level=0, **kwargs)



dummyAccf = Helper(window = 250)
dummyPccf = Helper(window = 250)
rollobj = roll(data, w = 250)#.apply(lambda x: rollccf(x))

rollobj.apply(lambda x: dummyAccf.append(rollaccf(x, maxLag=8)))
rollobj.apply(lambda x: dummyPccf.append(rollpccf(x, maxLag=2)))
    
data['accf'] = dummyAccf.result
data['pccf'] = dummyPccf.result


    
# def optimize(x, process, maxLag = None, args = None):
#  
#     x = x.reshape(1,250)
#     obj = process(x, p = args[0], q = args[1])
#      
#     return obj
#  
# print('\nstarting ARMApqN...\n')
#  
# dummy  = Helper(window = 250)
# data1['ARMApqN'] = data1['Return'].rolling(window = 250).apply(lambda x: dummy.append(optimize(x, logL.maxVARMApqN, args = (2,2)))) # VARMA now works also for 1-dim processes
# data1['ARMApqN'] = dummy.result
 


print(data)



















