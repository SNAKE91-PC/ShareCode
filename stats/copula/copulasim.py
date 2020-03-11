'''
Created on 2 Feb 2020

@author: snake91
'''

import numpy as np
import funcy
import scipy.optimize as spo
import matplotlib.pyplot as plt


# TODO: rename to biconditionalCopula
def conditionalCopula1(args):
    
    v, f, theta = args
    
    while True:
        
        u = np.random.uniform() # source
        
        def g(t,f,u,theta):
            
            return ( f(theta, u + 10e-7, t) - f(theta, u, t) )/ 10e-7
    
        g = funcy.rpartial(g, f = f, u = v, theta = theta)
    
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


def conditionalCopula2(args, f, theta):
    
    from algopy import UTPM

#     def g1(theta, f, u,t):
#     
#         a = UTPM.init_jacobian([u,t])
#         y = f(theta, a)
#         res = UTPM.extract_jacobian(y)#UTPM.extract_jacobian(y)
# 
#         return res[0]

    def g(theta, f, args):

        hes = UTPM.init_jacobian(args)
        y = f(theta, hes)
        res = UTPM.extract_jacobian(y)#UTPM.extract_jacobian(y)
        
        return res[1]
        
        
#     if len(args) == 3:
#         v, f, theta = args
#         g = g1
#     elif len(args) == 4:
#         v, b, f, theta = args
#         g = g2

    u = args[0]
    args = args[1:]
    while True:
        
#         u = np.random.uniform() # source

#         if len(args) == 3:
# #             g = funcy.rpartial(g,  u = v, theta = theta, f = f)
#             gtmp = lambda t: g(theta, f, v, t)
#             
            
#         elif len(args) == 4:
#             g = funcy.rpartial(g, f = f, theta = theta, u = v, t = b)
#         gtmp = lambda t: g(theta, f, args[:-1])
    

#         gtmp = lambda x: g(theta, f, args[:-1])
        
        try:
#             t = spo.brenth(lambda x: gtmp(x) - u, 10e-6, 1-10e-6, xtol = 10e-10)
            t = spo.brenth(lambda x: g(theta, f, [x] + list(args)) - u, 10e-6, 1-10e-6, xtol = 10e-10)
#             t = spo.newton_krylov(F = lambda x: g(x) - u, xin = (1-u))[0]
#             t = spo.minimize(fun = lambda x: (g(x) - u)**2, x0 = (u,), bounds = [(10e-7,1-10e-7)], tol = 10e-16).x[0]
        except Exception as e:
            print(e, args[0], u, theta)#, u, v, sep = " ")
#             plt.plot(np.linspace(10e-5,1-10e-5,1000), g(np.linspace(10e-5,1-10e-5,10000)))
#             v = np.random.uniform()
                
        else:
            break
    
    
    return (*args, t)


