'''
Created on 21 Feb 2020

@author: snake91
'''


import numpy as np
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import pathos.pools as pp
import pandas as pd
from findiff import FinDiff    
# from copula.copulafunc import clayton
import functools

from algopy import UTPM
import itertools as it

def clayton(theta, *x):
    
    x = x[0]
    return (x[0]**(-theta) + x[1]**(-theta)-1)**(-1/theta)



def gradient(f, theta, x):
    
    a = UTPM.init_hessian(x)
    y = f(theta, a)
    res = UTPM.extract_hessian(2,y)#UTPM.extract_jacobian(y)
    out = res.diagonal(1)
        
    return out[0]


def _prepare_data(u,v):

    udf = pd.DataFrame(u, columns = ['u'])
    udf['dummy'] = 1
    vdf= pd.DataFrame(v, columns = ['v'])
    vdf['dummy'] = 1
    
    t = pd.merge(udf, vdf,how ='inner', on = 'dummy')
    t = np.array(t[['u','v']])

    return t

def Gdensity(t,f,u,theta):
    # generic
    v = _prepare_data(t, u)
    
    grad = np.array(list(map(lambda x: gradient(f, theta, list(x)), v)))
#     st.norm.pdf(st.norm.ppf(t)) * st.norm.pdf(st.norm.ppf(u)) * \
#     return  (((f(theta, u + h, t + h) - f(theta, u, t + h)) / h  - (f(theta, u + h, t) - f(theta, u, t)) / h)  /h)
    return t * u * grad.reshape(tuple([u.shape[0], u.shape[0]]))


def Ndensity(u,v,rho):

    t = _prepare_data(u, v)
        
    a = (1/np.sqrt(slin.det(rho)))
    
     
    bfunc = lambda x,y: np.exp(-0.5 * \
                        np.dot(np.dot(np.array([st.norm.ppf(x), st.norm.ppf(y)]), (slin.inv(rho) - np.identity(2))), np.array([st.norm.ppf(x), st.norm.ppf(y)]).T))
     
    b = np.array(list(map(lambda x: bfunc(x[0], x[1]), t))).reshape(tuple([u.shape[0], u.shape[0]]))
    
    return st.norm.pdf(st.norm.ppf(u)) * st.norm.pdf(st.norm.ppf(v)) * a * b   
    
    
    
    



if __name__ == '__main__':
    
    c = 0
    
    import scipy.optimize as spo
    import scipy.stats as st
    import scipy.linalg as slin
    
    f = clayton

    v = np.linspace(0.000001,0.99999,100) 
    u = np.linspace(0.000001,0.99999,100)
    
    
    theta = 20
    
    Cxy = Gdensity(v,f,u,theta)

    U,V = np.meshgrid(u,v)
     
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(U, V, Cxy, 200, cmap = 'hot')
    fig.colorbar(cp)
    
    ###############################################

    rho = np.array([[1,0.2],[0.2, 1]])
        
    v = np.linspace(0.000001,0.99999,100) 
    u = np.linspace(0.000001,0.99999,100) 
    
    Cxy = Ndensity(u,v,rho)
    
    U,V = np.meshgrid(u,v)
    
    df = pd.DataFrame(Cxy)
    df.to_csv("/home/snake91/checks/copula/algo.csv")
    
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(st.norm.ppf(U), st.norm.ppf(V), Cxy, 200, cmap = 'hot')
    fig.colorbar(cp)

     
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(st.norm.ppf(V), st.norm.ppf(U), Cxy, 100, cmap = 'hot')
    fig.colorbar(cp)    
    
    
    














    
    
