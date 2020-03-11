'''
Created on 29 Feb 2020

@author: snake91
'''


import numpy as np
from copula.copulasim import conditionalCopula2


def clayton(theta, *x):
    
    if len(x)==1:
        x = x[0]
        
    return (x[0]**(-theta) + x[1]**(-theta)-1)**(-1/theta)



if __name__ == "__main__":
    
    f = clayton
    
    nvines = 4
    
    theta = np.random.randint(low = 1, high = 40, size = (nvines, nvines))
    
    n = 5
    wi = np.random.uniform(size = (nvines, n))
    vi = np.zeros(shape = (nvines, nvines, n))
    
    vi[0,0,:] = wi[0,:].copy()
    xi = np.zeros(shape = (nvines, n))
    xi[0,:] = wi[0,:].copy()
    
    for i in range(1, wi.shape[0]):
        
        vi[0,i,:] = wi[i,:].copy()
        
        for k in reversed(range(0, wi.shape[0])):
            
            vi[k,k,:] = wi[k,:].copy()
            
            args = vi[0,i,:], vi[k,k,:] 
            
            for q, couple in enumerate(zip(*args)):
                res = conditionalCopula2(couple, f, theta[k,i-k])
                
                print(res)
                vi[0,i,q] = res[1] 
                
        xi[i,:] = vi[0,i,:].copy()
        
        if i == (wi.shape[0]-1):
            continue
        
        for j in range(1, i):
            
            vi[i,j,:] = f(theta[i,i-j-1], vi[i,j-1,:], vi[j-1,j-1,:]) 
            
            
    print("")
        
        
        
        
        
        
        