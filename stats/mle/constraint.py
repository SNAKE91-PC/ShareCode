'''
Created on Dec 24, 2018

@author: snake91
'''

import numpy as np


def consMGARCH(y):
    
    return


# def consGARCHpq(y, p, q):
#     
#     return consARp(y[1:][p: (p+q)])

def consCorrMatrix(y):
    
    # check for positive definiteness
    return

def consARCHp(y):
    
    return consARp(y)

def consARp(y):
    
    I = np.diag(np.ones(shape = (1 * len(y))))[0: len(y)-1]
    F = np.vstack((y, I))
    
    eigenvalues, eigenvec = np.linalg.eig(F) # @UnusedVariable
    
    f = lambda x: (np.real(x)**2 + np.imag(x)**2 )  #<1
    
#     return len(eigenvalues) - np.sum(np.abs(f(eigenvalues)))

    return 1 - np.max(f(eigenvalues))


#     print 'constraint'
#     if all(f(eigenvalues)):
#         return 1
#     else:
#         return -1


def consMAq(y):
    
    return consARp(y)


def consVARp(phi, p):
    
    if p == 0:
        return 1
    
    nprocess = int(np.sqrt(len(phi)/p))
    phiList = []
    
    phi = list(phi)
#     eigList = []
    for i in range(nprocess**2, len(phi)+1, nprocess**2):
        phiMatrix = np.asmatrix(phi[i-nprocess**2: i]).reshape((nprocess, nprocess))
        phiList.append(phiMatrix)
        
    I = np.diag(np.ones(nprocess * p))
    
        
    F = np.vstack((np.hstack(phiList), I))
    zeros = np.zeros(shape = (F.shape[0], F.shape[0]-F.shape[1]))
    
    F = np.hstack((F, zeros))
    
    eigenvalues, eigenvectors = np.linalg.eig(F) # @UnusedVariable
    
    f = lambda x: (np.real(x)**2 + np.imag(x)**2) #< 1 
    
#     if 1 - np.max(np.abs(f(eigenvalues))) <0:
#         print('constraint')
    return 1 - np.max(f(eigenvalues)) #len(eigenvalues) - np.sum(np.abs(f(eigenvalues)))
#     if all(f(a)):
#         return 1
#     else:
#         print 'constraint'
#         return -1
    
    
def consVMAq(psi, q):
    
    return consVARp(psi, q)
    
    
    
    