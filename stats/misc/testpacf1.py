'''
Created on Dec 20, 2018

@author: snake91
'''


 
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from mle import likelihood as logL
import scipy.optimize as opt
import scipy.stats as st
import scipy.spatial as spa
import scipy.linalg as slin
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(r'/home/snake91/git/ShareCode/autocorr/DJI.csv')
data['Date'] = pd.to_datetime(data['Date'])

scenario = {}
scenario['Lehman'] = ('2007-04-01', '2008-12-31')

data = data[(data['Date'] > scenario['Lehman'][0]) & (data['Date'] < scenario['Lehman'][1] )]

data['Return'] = data['Close'] / data['Close'].shift() - 1
data.dropna(inplace = True)



acfMatrix = lambda lag: np.corrcoef(data['Return'].shift(lag).dropna(), data['Return'][lag:])#x[1,0]


def acvfMatrix(lag):
    
    
        vardiag = np.asmatrix(np.diag([
                                np.sqrt(np.var(data['Return'].shift(lag).dropna())),
                                np.sqrt(np.var(data['Return'][lag:]))
                                ]))
        acfcorr = np.asmatrix(acfMatrix(lag)) 
                   
                   
        return vardiag * acfcorr * vardiag
        
        

acf = lambda lag: acfMatrix(lag)[1,0]
acvf = lambda lag: np.array(acvfMatrix(lag))[1,0]

def pacf(lag):
#     acfList = [acf(i) for i in range(0,lag-1)]
    acvfList =[acvf(i) for i in range(0,lag)]
    acvfListMatrix =slin.toeplitz(acvfList)
    
    invacvfList = np.linalg.inv(acvfListMatrix)
    
    pacfList = np.matrix(invacvfList) * np.matrix([acvf(i) for i in range(1,lag+1)]).T
    
    return pacfList[-1, 0]


plt.plot(range(1,50), [pacf(i) for i in range(1,50) ])
plot_pacf(data['Return'].get_values(), lags = 10)
plot_acf(data['Return'].get_values(), lags = 10)


plot_copula = lambda lag: plt.scatter(data['Return'].shift(lag).dropna(), data['Return'][lag:], s=2.5)



plot_copula(2)

