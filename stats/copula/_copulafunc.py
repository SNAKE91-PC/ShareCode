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

# def clayton(theta, *args):
#     
#     # args are u,v...
#     
#     if len(args) == 1:
#         args = args[0]
#         
#     out = 0
# 
# #     args = list(map(lambda x: x[0] if (type(x) == np.ndarray and len(x[0]))==1 else x, args))    
#     for i in range(len(args)):
#         
#         out += (args[i]**(-theta))
#     
#     try:
#         out =  out**(-1/theta) #(u**(-theta) + v**(-theta) - 1)**(-1./theta)
#     except:
#         pass
#     
#     return out

    


