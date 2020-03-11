'''
Created on Dec 13, 2018

@author: snake91
'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import pathos.pools as pp
import pandas as pd

import functools
import scipy.optimize as spo
# from copula.copulasim import conditionalCopula
from copula.copulafunc import clayton
# TODO: rewrite with **kwargs instead of the explicit theta
# f = lambda u,v, theta: (u*v) / (u+v-u*v)
# 
# f = lambda u,v, theta: np.exp(- ( ( (-np.log(u))**theta + (-np.log(v))**theta )**(1./theta) )) #gumbel
# f = lambda u,v, theta: (-1./theta) * np.log(1 + (((np.exp(-theta * u) - 1) * 
#                                         (np.exp(-theta * v) - 1) )/ (np.exp(-theta) - 1)))
# f = lambda u,v, theta: (u**(-theta) + v**(-theta) - 1)**(-1./theta)  #clayton


# TODO: rename to biconditionalCopula
def conditionalCopulaContour(args):
    
    v, f, theta, q = args
    
    print(q)
    
    while True:
        
#         u = np.random.uniform() # source
        
        
#         def g(t,f,u,theta):
#             
#             return ( f(theta, u + 10e-7, t) - f(theta, u, t) )/ 10e-7
#     
#         g = functools.partial(g, f = f, u = v, theta = theta)
    
        try:
#             t = spo.brenth(lambda x: g(x) - 0.1, 10e-7, 1-10e-7, xtol = 10e-7)
            t = spo.brenth(lambda x: f(theta, x, v) - q, 10e-7, 1-10e-7, xtol = 10e-7)
#             t = spo.newton_krylov(F = lambda x: g(x) - u, xin = (1-u))[0]
#             t = spo.minimize(fun = lambda x: (g(x) - u)**2, x0 = (u,), bounds = [(10e-7,1-10e-7)], tol = 10e-16).x[0]
        except Exception as e:
            
            v = np.random.uniform()
#             print(e, u, v, sep = " ")
            
#             v = np.random.uniform()
                
        else:
            break
    
    
    return (v, t, q)
        


if __name__ == '__main__':
    

    f = clayton
    v = np.linspace(0.1, 0.9,10000)#np.random.uniform(size = 10000) #target

    pool = pp.ProcessPool(4)
    
    theta = 5
    
#     Q = np.linspace(0.1, 0.9, 10)
    q = 0.9
#     for q in Q: 
    data = list(map(lambda x: tuple([x, f, theta, q]), v)) 
    copulaList = pool.map(conditionalCopulaContour, data)
    
    x = np.array(list(map(lambda x: x[0], copulaList)))
    y = np.array(list(map(lambda x: x[1], copulaList)))

    plt.scatter(x,y, label = 'q ' + str(round(q,2)), s= 1.2)
    plt.legend()

    plt.show()
    
#     plt.figure()
#     plt.scatter(x,y, s=0.1)
    
    C = f(x,y,theta)    
    
    
    data = pd.DataFrame({'x': x, 'y': y, 'C': C})
    data.to_csv("/home/snake91/data.csv", index = False)
    
    print("done")
#     plt.scatter(x,z, s = 0.1)
#     
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax = fig.gca(projection='3d')
#      
#     ax.scatter(x, y, z, s=0.75)
    
    
    
    
    
    
    
    
