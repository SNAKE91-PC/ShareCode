'''
Created on 2 Feb 2020

@author: snake91
'''

import numpy as np
import functools 
import scipy.optimize as spo



# TODO: rename to biconditionalCopula
def conditionalCopula(args):
    
    v, f, theta = args
    
    while True:
        
        u = np.random.uniform() # source
        
        def g(t,f,u,theta):
            
            return ( f(theta, u + 10e-7, t) - f(theta, u, t) )/ 10e-7
    
        g = functools.partial(g, f = f, u = v, theta = theta)
    
        try:
            t = spo.brenth(lambda x: g(x) - u, 10e-7, 1-10e-7, xtol = 10e-7)
#             t = spo.newton_krylov(F = lambda x: g(x) - u, xin = (1-u))[0]
#             t = spo.minimize(fun = lambda x: (g(x) - u)**2, x0 = (u,), bounds = [(10e-7,1-10e-7)], tol = 10e-16).x[0]
        except Exception as e:
            print(e, u, v, sep = " ")
            
            v = np.random.uniform()
                
        else:
            break
    
    
    return (v, t)