'''
Created on Dec 8, 2018

@author: snake91
'''

import scipy.stats as st
import numpy as np
import pandas as pd

def iidHist(y, window = 250, q = 0.99):
    
    y = pd.Series(y)
    yUpper = y.rolling(window = window).apply(lambda x : np.percentile(x, q))
    yLower = y.rolling(window = window).apply(lambda x : np.percentile(x, 1-q))
    yMean = y.rolling(window = window).apply(lambda x : np.mean(x))
    
    return [np.array(yUpper), yMean, np.array(yLower)]
   
    
def arpGaussian(y, phi = [0.1], window = 250, q = 0.99):
    
    assert( window <= len(y)-len(phi) )
    
    yUpperList = []
    yLowerList = []
    
    eList = []
    yMean = []
    for i in range(len(y)):

        yPred = 0.
        
        for p in range(1, len(phi) + 1):
        
            yPred += phi[p-1] * y[i-p]
            
        eps = y[i] - yPred
        
        yMean.append(yPred)
        eList.append(eps)
        
    sigma = np.sqrt(np.var(eList))
    
    for t in range(0, len(y)):
        u = 0
        for n in range(1, len(phi) + 1):
            
            u += phi[n-1] * y[t-n] 
            
        yUpper = u + sigma**2 * st.norm.ppf(q = q) 
        yLower = u + sigma**2 * st.norm.ppf(q = 1-q)
        
        yUpperList.append(yUpper)
        yLowerList.append(yLower)
            
    yList = [yUpperList, yMean, yLowerList]
    
    return yList


def maqGaussian(y, psi = [0.1], window = 250, q = 0.99):
    
    
    assert( window <= len(y) - len(psi))
    
    yUpper = []
    yLower = []
    
    eList = []
    yMean = []
    for i in range(len(y)):
        
        yPred = 0.
          
        for q in range(1, len(psi) + 1):
            
            yPred += psi[q-1] * eList[i-q] 
            
        eps = y[i] - yPred
        
        yMean.append(yPred)
        eList.append(eps)
        
    sigma = np.sqrt(np.var(eList))
    
    yUpperList = []
    yLowerList = []
    
    for t in range(0, len(y)):
        u = 0
        for n in range(1, len(psi) + 1):
            
            u += psi[n-1] * y[t-n] 
            
        yUpper = u + sigma**2 * st.norm.ppf(q = q) 
        yLower = u + sigma**2 * st.norm.ppf(q = 1-q)
        
        yUpperList.append(yUpper)
        yLowerList.append(yLower)
        
    return [yUpperList, yMean, yLowerList]



















