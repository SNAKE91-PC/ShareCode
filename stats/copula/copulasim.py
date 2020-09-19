'''
Created on 2 Feb 2020

@author: snake91
'''

import numpy as np
# import funcy
import scipy.optimize as spo
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd

import tensorflow as tf

# TODO: rename to biconditionalCopula


# TODO: REWRITE USING TENSORFLOW DERIVATIVES
def conditionalCopula1(args):
    
    v, f, theta = args
    
    while True:
        
        u = np.random.uniform() # source
        
        def g(t,f,u,theta):
            
            return ( f(theta, u + 10e-7, t) - f(theta, u, t) )/ 10e-7
    
        g = partial(g, f = f, u = v, theta = theta)
    
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


def conditionalCopula2(args):
    
    v1, q = args[0]
    f, theta = args[1:] # v1, quantile 

    def g(v2, v1, theta, f):

        with tf.GradientTape(persistent=True) as g:
            g.watch([v1])
            z = f(theta, v1, v2)

        dy_da = g.gradient(z, v1)  
        
        return dy_da.numpy()

#     while True:
        
    try:
#         t = g(tf.constant(v2, dtype = tf.float32), tf.constant(v1, dtype = tf.float32), theta, f) - q
        t = spo.brenth(lambda v2: g(tf.constant(v2, dtype = tf.float32), tf.constant(v1, dtype = tf.float32), theta, f) - q, 10e-6, 1-10e-6, xtol = 10e-10)
#         t = spo.minimize(lambda v2: g(tf.constant(v2, dtype = tf.float32), tf.constant(v1, dtype = tf.float32), theta, f) - q, 10e-6, 1-10e-6, xtol = 10e-10)
    except ValueError as e:
        print(e, q, v1, theta)#, u, v, sep = " ")
        
        inputs = list(args[0][0:-1]) + [None]
        q = args[0][-1]
        func = args[1:]
    
        return (inputs, q, func)
#     else:
#         print("impossible")
#         break
    
    
    inputs = list(args[0][0:-1]) + [t]
    q = args[0][-1]
    func = args[1:]
    
    return (inputs, q, func)#(*args, t)


