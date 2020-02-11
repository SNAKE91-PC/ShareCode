'''
Created on 31 Jan 2020

@author: snake91


'''

'''
Created on Apr 15, 2019

@author: snake91
'''

'''
Created on Dec 13, 2018

@author: snake91
'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import scipy.optimize as spo
import functools
import pathos.pools as pp
import pandas as pd

from copula.copulamle import copulamle
from copula.copulasim import conditionalCopula
from copula import copulafunc
    
        


def partialtheta(f, theta1, theta2):
        
    np.random.seed(100)

    v = np.random.uniform(size = 10000) #target


    pool = pp.ProcessPool(4, servers = ['localhost:9090'])
    
    dataX = list(map(lambda x: tuple([x, f, theta1]), v))
    copulaListX = pool.map(conditionalCopula, dataX)
    
    x = np.array(list(map(lambda x: x[0], copulaListX)))
    y = np.array(list(map(lambda x: x[1], copulaListX)))
    
    Cxy = f(theta1, x,y)    
    
    dataXY = list(map(lambda x: tuple([x, f, theta2]), Cxy))
    copulaList = pool.map(conditionalCopula, dataXY)
    
    z = np.array(list(map(lambda x: x[1], copulaList)))
    
    Cxyz = f(theta2, Cxy, z) 

    # computing partial XZ
    
    
    def partialXZ(theta1, x,z,y,theta2):
        
        return f(theta2, f(theta1, x,z),y)
        
    def partialYZ(theta1, x,z,y,theta2):
        
        return f(theta2, f(theta1, y,z),x)
        
    
    args = x,y,Cxy
    mlethetaXY = spo.minimize(copulamle, 
                              x0 = (2,), 
                              #bounds = [(1, None)], 
                              args = (f, args), 
                              method='BFGS',
                              options={'gtol': 1e-12, 
                                       'eps': 1e-12,#.4901161193847656e-08, 
                                       'maxiter': None, 
                                       'disp': False
                                       })
    
    
#     zerothetaXZ = spo.brenth(lambda theta1: np.mean((partialXZ(theta1, x,z,y,theta2) - Cxyz)**2), 10e-2, 50, xtol = 10e-7)
    zerothetaXZ = spo.minimize(lambda theta1: np.mean((partialXZ(theta1, x,z,y,theta2) - Cxyz)**2), 
                               x0 = (2,), 
                               #bounds = [(1, 50)], 
                               method='BFGS',
                               options={'gtol': 1e-12, 
                                       'eps': 1e-12,#.4901161193847656e-08, 
                                       'maxiter': None, 
                                       'disp': False
                                       }) 
    
#     zerothetaYZ = spo.brenth(lambda theta1: np.mean((partialYZ(theta1, x,z,y,theta2) - Cxyz)**2), 10e-2, 50, xtol = 10e-7)
    zerothetaYZ = spo.minimize(lambda theta1: np.mean((partialYZ(theta1, x,z,y,theta2) - Cxyz)**2), 
                               x0 = (2,), 
                               #bounds = [(1, 50)], 
                               method='BFGS',
                               options={'gtol': 1e-12, 
                                       'eps': 1e-12,#.4901161193847656e-08, 
                                       'maxiter': None, 
                                       'disp': False
                                       })
 
    args = x,y,z,Cxyz
    mlethetaXYZ = spo.minimize(copulamle, 
                               x0 = (2,), 
                               #bounds = [(1, None)], 
                               args = (f, args), 
                               method='BFGS', 
                               options={'gtol': 1e-12, 
                                       'eps': 1e-12,#.4901161193847656e-08, 
                                       'maxiter': None, 
                                       'disp': False
                                       })
    
#         plt.scatter(x,y, s = 0.75,  color = 'blue')
#         plt.scatter(Cxy, z, s =0.75, color = 'red')
#         
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax = fig.gca(projection='3d')
#           
#         ax.scatter(x, y, z, s=0.75)
    
    return (mlethetaXY, zerothetaXZ, zerothetaYZ, mlethetaXYZ) 
    
    
    
if __name__ == '__main__':


    f = copulafunc.clayton

    theta1 = np.arange(5,45,1)
    theta2 = np.arange(5,45,1)
    
    
    out = []
    for t1 in theta1:
        for t2 in theta2:    
            mlethetaXY, zerothetaXZ, zerothetaYZ, mlethetaXYZ = partialtheta(f, t1, t2)
            
            out.append((mlethetaXY,zerothetaXZ,zerothetaYZ,mlethetaXYZ))
            
            print(t1,t2, sep = " ")
            
    mlethetaXY = list(map(lambda x: x[0].x, out))
    zerothetaXZ = list(map(lambda x: x[1].x, out))
    zerothetaYZ = list(map(lambda x: x[2].x, out))
    mlethetaXYZ = list(map(lambda x: x[3].x, out))
    print("")
            
    
    
    
    
    
    
