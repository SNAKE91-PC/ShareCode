'''
Created on 2 Feb 2020

@author: snake91
'''

import numpy as np

# TODO: rewrite with **kwargs instead of the explicit theta

import warnings
 
warnings.filterwarnings('error')


def gumbel(u,v, theta):
    
    return np.exp(- ( ( (-np.log(u))**theta + (-np.log(v))**theta )**(1./theta) )) 

def frank(u,v,theta):

    return (-1./theta) * np.log(1 + (((np.exp(-theta * u) - 1) * 
                                        (np.exp(-theta * v) - 1) )/ (np.exp(-theta) - 1)))

def clayton(theta, *args):
     
    # args are u,v...
     
    out = 0
 
    for i in range(len(args)):
        
#         try:
        tmp = args[i]**(-theta)
        out +=tmp
#         except:
#             pass
            
    try:
        out =  out**(-1/theta) #(u**(-theta) + v**(-theta) - 1)**(-1./theta)
    except:
        pass
     
    return out

    


