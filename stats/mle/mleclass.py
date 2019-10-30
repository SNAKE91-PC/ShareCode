'''
Created on Dec 29, 2018

@author: snake91
'''

from mle import likelihood as logL
from mle import simulate as sim

import scipy.optimize as opt
import numpy as np
from statsmodels.tsa.stattools import acf, pacf, ccf
from warnings import warn


from mle import constraint as cons
# from mle.tests.VARFIMApdq import pMatrix



class mleobj(object):
    
    def __init__(self, classmodel = None, **kwargs):
        
        #TODO: might be a factory wrt classmodel

        self.classmodel = classmodel
        
        self.bestguess = False
        
        
#         if self.process is not None:
            
#             self.process = process
#             self.nprocess = process.shape[0]
        self.params = None
        
        if 'order' not in kwargs.keys():
            self.bestguess = True
            
#             self.pLag = 0
#             self.qLag = 0
                 
        if 'order' in kwargs.keys():
            self.params = kwargs['order']
        
            self.pLag = self.params[0]
            self.qLag = self.params[1]
#         else:
#     
#             warn("Process not assigned in the constructor")
#             
#             if 'order' in kwargs:
#                 
#                 params = kwargs['order']
#                 self.pLag = params[0]
#                 self.qLag = params[1]
#                 
#             else:
#                 raise Exception("No order neither process set")
#             
            
        
        
    
    def fit(self, process, **kwargs):
        
#         if self.process is None:
#             raise Exception("Process not assigned in the constructor")
        
        if self.classmodel == 'ARMA':
            
            if 'tol' in kwargs:
                tol = kwargs['tol']
            else:
                tol = 10e-16
                
            nprocess = process.shape[0]
            
            if self.params is None:
                
                if 'maxLag' in kwargs.keys():
                    maxLag = kwargs['maxLag']
                else:
                    maxLag = 5
                    
                self._bestguess(process, maxLag) 
            
            #TODO: add function that takes care of building distribution-specific args
            
            params = logL.maxVARMApqN(process, self.pLag, self.qLag)
        
        return params
    
    def simulate(self, t = 1000, params = None, shocks = None):
        
        if self.classmodel == 'ARMA':
            
            try:
                pMatrix = params['pMatrix']
            except KeyError:
                pMatrix = None
            
            try:
                qMatrix = params['qMatrix']
            except KeyError:
                qMatrix = None
                
            X = sim.varmapqGaussian(t, pMatrix, qMatrix, shocks)
            
        
        return X
    
    @staticmethod
    def _accfMatrix(process, maxLag):

        import itertools
        
        process = np.array(process).T
        comb = list(itertools.product(process, repeat = process.shape[0]))
        size = int(np.sqrt(len(comb)))

        u = np.zeros(shape = (size**2, maxLag))

        if len(comb) == 1:
            u[0] = acf( comb[0][0] )[:maxLag]
        else:
            for i in range(len(comb)):
                u[i] = ccf( comb[i][0], comb[i][1] )[:maxLag]  

        arr = np.array([np.reshape(u[:,i], newshape = (size,size)) for i in range(maxLag)])
        
#         u = np.reshape(u, newshape = ())

        return arr
    
    @staticmethod
    def _pccfMatrix(process, maxLag):
        
        import itertools
        
        return
    
    
    @staticmethod    
    def _bestguess(self, process, maxLag = 5):
        
        # computing cross-correlation matrix
        u = self._accfMatrix(process,  maxLag)
        v = self._pccfMatrix(process, maxLag)

        raise NotImplementedError
        ### automatic model identification
        
        return 
    
    
    
    
    