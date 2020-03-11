'''
Created on 11 Feb 2020

@author: snake91
'''

import numpy as np
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import pathos.pools as pp
    
from findiff import FinDiff    
# from copula.copulafunc import clayton
import functools



def clayton(theta, t,u):
    
    
    return (t**(-theta) + u**(-theta) -1 )**(-1/theta)

def Gdensity(t,f,u,theta):
    # generic
    
    h = 10e-6
    return t*u * \
      (((f(theta, u + h, t + h) - f(theta, u - h, t + h)) / (2 * h)  - (f(theta, u + h, t - h) - f(theta, u - h, t - h)) / (2 * h))  /(2 * h))
    


def Ndensity(t,u,rho):
    
    global c
    c+=1
    
#     a = (1/np.sqrt(slin.det(rho)))
#     
#     b = np.exp(-0.5 * \
#                         np.dot(np.dot(np.array([st.norm.ppf(t), st.norm.ppf(u)]), (slin.inv(rho) - np.identity(2))), np.array([st.norm.ppf(t), st.norm.ppf(u)]).T))
#     
#     print(c, st.norm.pdf(st.norm.ppf(t)) * st.norm.pdf(st.norm.ppf(t)) * a * b)
#     
#     
#     
#     return st.norm.pdf(st.norm.ppf(t)) * st.norm.pdf(st.norm.ppf(t)) * a * b   
    
    
    
    



if __name__ == '__main__':
    
    c = 0
    
    import scipy.optimize as spo
    import scipy.stats as st
    import scipy.linalg as slin
    import pandas as pd
    
    f = clayton
    v = np.linspace(0.00001,0.9999,100) #target
    u = np.linspace(0.00001,0.9999,100) #target
    
    copulaList = []
    
    theta = 5

    U,V = np.meshgrid(u,v)
    
    Cxy = Gdensity(U,f,V,theta)
    
    df = pd.DataFrame(Cxy)
    df.to_csv("/home/snake91/checks/copula/incratio.csv")
    
#     rho = np.array([[1,0.5],[0.99,1]])
#     Cxy = list(map(lambda y: list(map(lambda x: Ndensity(x,y,rho), u)), v))
    
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(U, V, np.array(Cxy), 100, cmap = 'hot')
    fig.colorbar(cp)

     
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(st.norm.ppf(V), st.norm.ppf(U), Cxy, 100, cmap = 'hot')
    fig.colorbar(cp)    
    
    
    














    
    
