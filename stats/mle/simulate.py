'''
Created on Nov 18, 2018

@author: snake91
'''


import numpy as np
import scipy.special as sp
from copy import deepcopy
import matplotlib.pyplot as plt

flatten = lambda l: [item for sublist in l for item in sublist]



def sim_markovchain(t, pmatrix, startvalue):
    
    """    
        pmatrix = {0: [0.8, 0.2], 1 : [0.4, 0.6]}
    
    """

    ### validate input
    
    if startvalue not in pmatrix.keys():
        print("startvalue not in possible states")
        raise Exception

    for state in pmatrix.keys():
        prob = pmatrix[state]
        if sum(prob) != 1:
            print("transition probabilities for state " + str() + " don't sum to 1")
            raise Exception
        else:
            pass
        
    #### input validated

    sample = [startvalue]
    a = np.arange(0, len(pmatrix.keys()))
    for idx in range(1, t):
        
        value =  np.random.choice(a, size = 1, p = pmatrix[sample[idx-1]])[0]
        sample.append(value)
        
    return np.array(sample)


def msiidN(t, transmat, startstate, paramsmean, paramsvar):
    
    sample_tr =  sim_markovchain(t, pmatrix = transmat, startvalue = startstate)
    sample = [0]

    
    for i in range(1, t):
        value = np.random.normal(paramsmean[sample_tr[i]], paramsvar[sample_tr[i]])
        sample.append(value)
    
    return sample



# def msarmaN(t, pmatrix, startstate, paramsp = None, paramsq = None):
#  
#     sample_tr =  sim_markovchain(t, pmatrix = pmatrix, startvalue = startstate)
#     sample = [0]
#     
#     maxord = max(list(map(lambda x: len(x), paramsp)) + list( map(lambda x: len(x), paramsp)) )
#     for i in range(maxord, t):
#         
#         if paramsp is not None and paramsq is not None:
#             value = armapqGaussian(2, phi = paramsp[sample_tr[i]], psi = paramsq[sample_tr[i]], y0 = [sample[i-maxord]])
#         if paramsp is None and paramsq is not None:
#             value = armapqGaussian(2, phi = None, psi = paramsq[sample_tr[i]], y0 = [sample[i-maxord]])
#         if paramsp is not None and paramsq is None:
#             value = armapqGaussian(2, phi = paramsp[sample_tr[i]], psi = None, y0 = [sample[i-maxord]])
#         sample.append(value[-1])
#      
#     return sample



def mgarcheccc(t, **kwargs):
    
    # contains mgarchccc and garch as special cases
    
    a0 = np.array(kwargs['a0'])
    alphaMatrix = np.array(kwargs['alphaMatrix'])
    betaMatrix = np.array(kwargs['betaMatrix'])
    
#     assert(all(a0) > 0)
#     assert(all(alphaMatrix) > 0)
#     assert(all(betaMatrix) > 0)
    
    corrMatrix = kwargs['corrMatrix']
    
    naorder = len(alphaMatrix)
    nborder = len(betaMatrix)
    # there should be other checks (e.g. alphaMatrix should be square)

    idx = max(naorder, nborder)
    
    dim = alphaMatrix[0].shape[0] # they must have all the same dimensions anyway
    
    chol = np.linalg.cholesky(corrMatrix)
    
    if 'shocks' not in kwargs.keys():
        shocks = np.random.normal(size = (dim, t))
    else:
        shocks = kwargs['shocks']
        # there should be an assert here
        
    shocks = np.array(np.dot(chol, shocks))
    if 'sigma0' not in kwargs.keys():
        sigma0 = np.sqrt(np.random.normal(size = (dim, idx))**2)
    else:
        sigma0 = kwargs['sigma0']
        assert(len(sigma0) == len(alphaMatrix))
    
    
    Y = np.zeros(shape = (dim, t))
    SIGMA = np.zeros(shape = (dim, t))
    
    SIGMA[:, 0: idx] = sigma0
    
    
    for i in range(idx, t):
        
        variance = np.zeros(shape = a0.shape)#deepcopy(a0)
        variance += a0
        
        for a in range(1, len(alphaMatrix) + 1):
        
            variance +=  np.dot(alphaMatrix[a-1], Y[:, i-a : i-a + 1]**2)

        for b in range(1, len(betaMatrix) + 1):
            
            variance += np.dot(betaMatrix[b-1], SIGMA[:, i-b : i-b + 1]**2)
            
        sigma = np.sqrt(variance)
        
        garchErrors = shocks[:, i: i+1] * sigma

        SIGMA[:, i : i+1 ] = sigma
        Y[:, i: i+1 ] = garchErrors
        
        
#     Y = np.dot(chol, Y)
    
    return np.asmatrix(Y)
    
    



def mgarchccc(t, **kwargs):

    a0    = np.array(kwargs['a0'])     
    alpha = np.array(kwargs['alpha'])
    beta  = np.array(kwargs['beta'])
    
    assert(all(a0)    > 0), "all alpha0 must be positive" #study a = 0, decreasing variance
    assert(all(alpha) > 0), "all alpha must be positive"
    assert(all(beta)  > 0), "all beta must be positive"

    corrMatrix = kwargs['corrMatrix']
    eigv, eigm = np.linalg.eig(corrMatrix) #@Unusedvariable
    
    assert(all(eigv) > 0), "correlation matrix is not positive definite"
    
    naorder = alpha.shape[1]
    nborder = beta.shape[1]
    
    assert(len(alpha) == len(beta) == corrMatrix.shape[0])    
    
    dim = len(alpha)
    
    chol = np.linalg.cholesky(corrMatrix)
    shocks = np.random.normal(size = (dim, t))
    
#     shocks = chol * shocks
    
    if 'sigma0' not in kwargs.keys():
        sigma0 = np.random.normal(size = (dim, max(naorder, nborder)))
    else:
        sigma0 = kwargs['sigma0']
        assert(len(sigma0) == alpha.shape[1])
    
    Y = np.zeros(shape = (dim, t))
     
    for process in range(dim):
        y = garchpqGaussian(t, sigma0 = sigma0[process], 
                            a0 = a0[process][0], 
                            alpha = alpha[process], 
                            beta = beta[process], 
                            shocks = shocks[process]) 
        
        Y[process, :] = y

    
    Y = chol * Y    

    return Y
     

def mgarchdcc(t, **kwargs):
    
    return

def mgarchvcc(t, **kwargs):
    
    return



def garchpqGaussian(t, **kwargs):
    '''
        sigma0    optional, initial values for sigma
        a0        required, a0 ARCH
        alpha     required, a  ARCH
        beta      required, b  GARCH
        shocks    optional, default N(0,1)
    '''
    alpha = kwargs['alpha']
    beta = kwargs['beta']

    #np.random.seed(1)
    
    idx = max(len(alpha), len(beta))
        
    if 'sigma0' not in kwargs.keys():
        sigma0 = np.sqrt(np.random.normal(size = len(alpha))**2)
    else:
        sigma0 = kwargs['sigma0']
        assert(len(sigma0) == len(alpha))
        
    a0 = kwargs['a0']
    
    assert(a0 >= 0) #study a = 0, decreasing variance
    assert(all(map(lambda x: x >= 0, alpha)))
    
    if 'shocks' not in kwargs.keys():
        stdErrors = np.random.normal(size = t)
    else:    
        stdErrors = np.random.normal(size = t)
    
    
    yList = list(sigma0)
    sigmaList = list(sigma0**2)
     
    for i in range(idx, t):
        
        variance = deepcopy(a0)

        for a in range(1, len(alpha) + 1):
        
            variance +=  alpha[a-1] * yList[i-a]**2

        for b in range(1, len(beta) + 1):
            
            variance += beta[b-1] * sigmaList[i-b]**2
            
        sigma = np.sqrt(variance)
        sigmaList.append(np.float(sigma))
        
        garchErrors = stdErrors[i] * sigma
        yList.append(np.float(garchErrors))
        
    return np.array(yList)



def archpGaussian(t, **kwargs):#, y0 = [0.]):

    alpha = kwargs['alpha']
    
    if 'sigma0' not in kwargs.keys():
        sigma0 = list(np.random.normal(size = len(alpha)))
        
    else:
        assert(len(sigma0) == len(alpha))
        
    a0 = kwargs['a0']
    
    assert(a0 >= 0) #study a = 0, decreasing variance
    assert(all(map(lambda x: x >= 0, alpha)))
    
    
    stdErrors = np.random.normal(size = t)
    
    yList = sigma0
#     sigmaList = [a0]
     
    for i in range(len(alpha), t):
        
        variance = a0

        for a in range(1, len(alpha) + 1):
        
            variance +=  alpha[a-1] * yList[i-a]**2

        sigma = np.sqrt(variance)
#         sigmaList.append(sigma)
        archErrors = stdErrors[i] * sigma
        
        yList.append(archErrors)
        
    return np.array(yList)
    


def varfimapdqGaussian(t, pMatrix, qMatrix, dcoeff, y0):

    # each p should have its pMatrix A
    # A should be always squared
    nprocesses = y0.shape[0] #y
    nporder    = len(pMatrix)
    nqorder    = len(qMatrix)
    nintorder  = max(dcoeff)
    
    assert(y0.shape[0] == len(pMatrix[0]))
    assert(y0.shape[0] == len(qMatrix[0]))
    assert(y0.shape[0] == len(dcoeff))
    
    shocks = np.asmatrix(np.random.normal(size = (nprocesses, t)))
    
    yList = np.asmatrix(np.zeros(shape = (nprocesses, t)))
    yList[:, 0: nporder] = y0#.T
    
    binom = []
    
      
    
    
    for coeff in dcoeff:
        binomcoeff = np.array([sp.gamma(j-coeff) / (sp.gamma(j+1) * sp.gamma(-coeff)) for j in range(0, t)])
        binomcoeff[np.isnan(binomcoeff)] = 0.
        binom.append(binomcoeff)
        
    
    startIdx = max([nporder, nqorder])
    
    # AR
    for i in range(startIdx, t):
        
        y = np.zeros(shape = (nprocesses, 1))
        
        for p in range(1, nporder + 1):
            
            y += pMatrix[p-1] * yList[:, i-p]
                
        y +=  shocks[:, i]
        
        yList[:, i] += y
        
    # I
    for i in range(startIdx, t):
        
        y = np.zeros(shape = (nprocesses, 1))
        
        c = 0
        for dlist in binom:
            
            for d in range(1, i+1): #len(dlist)
                y[c] += -dlist[d] * yList[c, i-d]
                
            c+=1
            
        yList[:, i] += y
        
    # MA
    for i in range(startIdx, t):
        
        y = np.zeros(shape = (nprocesses, 1))
        
        for q in range(1, nqorder + 1):
            
            y += qMatrix[q-1] * shocks[:, i-q]

        
        
        yList[:, i] += y
        
        
    return yList












def varimapdqGaussian(t, pMatrix, qMatrix, dcoeff, y0):

    # each p should have its pMatrix A
    # A should be always squared
    nprocesses = y0.shape[0] #y
    nporder    = len(pMatrix)
    nqorder    = len(qMatrix)
    nintorder  = max(dcoeff)
    
    assert(y0.shape[0] == len(pMatrix[0]))
    assert(y0.shape[0] == len(qMatrix[0]))
    assert(y0.shape[0] == len(dcoeff))
    
    shocks = np.asmatrix(np.random.normal(size = (nprocesses, t)))
    
    yList = np.asmatrix(np.zeros(shape = (nprocesses, t)))
    yList[:, 0: nporder] = y0#.T
    
    binom = []
    
    for coeff in dcoeff:
        binomcoeff = [(-1)**d * sp.binom(coeff, d) for d in range(0, coeff+1)]
        binom.append(binomcoeff)
        
    
    startIdx = max([nporder, nqorder, nintorder])
    
    # AR
    for i in range(startIdx, t):
        
        y = np.zeros(shape = (nprocesses, 1))
        
        for p in range(1, nporder + 1):
            
            y += pMatrix[p-1] * yList[:, i-p]
                
        y +=  shocks[:, i]
        
        yList[:, i] += y
        
    # I
    for i in range(startIdx, t):
        
        y = np.zeros(shape = (nprocesses, 1))
        
        c = 0
        for dlist in binom:
            
            for d in range(1, len(dlist)):
                y[c] += -dlist[d] * yList[c, i-d]
                
            c+=1
            
        yList[:, i] += y
        
    # MA
    for i in range(startIdx, t):
        
        y = np.zeros(shape = (nprocesses, 1))
        
        for q in range(1, nqorder + 1):
            
            y += qMatrix[q-1] * shocks[:, i-q]

        
        
        yList[:, i] += y
        
        
    return yList






def varmapqGaussian(t, pMatrix = None, qMatrix = None, shocks = None, y0 = None):#, y0):

    # each p should have its pMatrix A
    # A should be always squared
    
    if pMatrix is None:
        dim = len(qMatrix[0])
        nporder = 0
        nqorder = len(qMatrix)
    
    elif qMatrix is None:
        dim = len(pMatrix[0])
        nporder = len(pMatrix)
        nqorder = 0
    
    elif pMatrix is None and qMatrix is None:
        raise Exception("Either pMatrix or qMatrix must be valued")
    
    else:
        dim = len(qMatrix[0])
        nporder    = len(pMatrix)
        nqorder    = len(qMatrix)
    
    if y0 is None:  
        y0 = np.asmatrix(np.random.normal(size = (1, dim))).T
    
    
    nprocesses = dim
    
    if shocks is None:
        shocks = np.asmatrix(np.random.normal(size = (nprocesses, t)))
    else:
        assert(shocks.shape == (nprocesses, t))

    yList = np.asmatrix(np.zeros(shape = (nprocesses, t)))
    yList[:, 0: max(nporder,nqorder)] = y0#.T
    
    for i in range(max(nporder, nqorder), t):
        
        y = np.zeros(shape = (nprocesses, 1))
        
        for p in range(1, nporder + 1):
            
            y += pMatrix[p-1] * yList[:, i-p] 

        for q in range(1, nqorder + 1):
            
            y += qMatrix[q-1] * shocks[:, i-q]

        y +=  shocks[:, i]
        
        yList[:, i] += y

        
    return yList



def varpGaussian(t, pMatrix, y0):

    # each p should have its pMatrix A
    # A should be always squared
    nprocesses = y0.shape[0] #y
    nporder    = len(pMatrix)
    
    assert(y0.shape[0] == len(pMatrix[0]))
    
    shocks = np.asmatrix(np.random.normal(size = (nprocesses, t)))
    
    yList = np.asmatrix(np.zeros(shape = (nprocesses, t)))
    yList[:, 0: nporder] = y0#.T
    for i in range(nporder, t):
        
        y = np.zeros(shape = (nprocesses, 1))
        
        for p in range(1, nporder + 1):
            
            y += pMatrix[p-1] * yList[:, i-p] 

        y +=  shocks[:, i]
        
        yList[:, i] = y
        
        
    return yList











def vmaqGaussian(t, qMatrix, y0):

    # each p should have its pMatrix A
    # A should be always squared
    nqrocesses = y0.shape[0] #y
    nqorder    = len(qMatrix)
    
    assert(y0.shape[0] == len(qMatrix[0]))
    
    shocks = np.asmatrix(np.random.normal(size = (nqrocesses, t)))
    
    yList = np.asmatrix(np.zeros(shape = (nqrocesses, t)))
    yList[:, 0: nqorder] = y0#.T
    for i in range(nqorder, t):
        
        y = np.zeros(shape = (nqrocesses, 1))
        
        for q in range(1, nqorder + 1):
            
            y += qMatrix[q-1] * shocks[:, i-q] 

        y +=  shocks[:, i]
        
        yList[:, i] = y
        
        
    return yList







def vma1Gaussian(t, qMatrix, y0):

    # each p should have its pMatrix A
    # A should be always squared

    nprocesses = qMatrix.shape[0]
    
    shocks = np.asmatrix(np.random.normal(size = (nprocesses, t)))
    
    yList = np.asmatrix(np.zeros(shape = (nprocesses, t)))
    yList[:, 0] = y0
    
    for i in range(1,t):
        
        y = qMatrix * shocks[:, i-1] + shocks[:, i]
        
        yList[:, i] = y
    
    
    return yList


def var1Gaussian(t, pMatrix, y0):

    # each p should have its pMatrix A
    # A should be always squared
    nprocesses = pMatrix.shape[0] #y
    
    shocks = np.asmatrix(np.random.normal(size = (nprocesses, t)))
    
    yList = np.asmatrix(np.zeros(shape = (nprocesses, t)))
    yList[:, 0] = y0#.T
    for i in range(1, t):
        #y = np.zeros(size = (nprocesses, 1))
        y = pMatrix * yList[:, i-1] + shocks[:, i]

        yList[:, i] = y
        
        
    return yList




def var1Student(t, pMatrix, y0, df = 10):

    # each p should have its pMatrix A
    # A should be always squared
    nprocesses = pMatrix.shape[0] #y
    
    shocks = np.asmatrix(np.random.standard_t(size = (nprocesses, t), df = df))
    
    yList = np.asmatrix(np.zeros(shape = (nprocesses, t)))
    yList[:, 0] = y0#.T
    for i in range(1, t):
        #y = np.zeros(size = (nprocesses, 1))
        y = pMatrix * yList[:, i-1] + shocks[:, i]

        yList[:, i] = y
        
        
    return yList









def arfimapdqGaussian(t, phi = [0.1], psi = [0.1], dcoeff = 0,):
     
    y0 = list(np.zeros(max(len(phi), len(psi))))
    
    shocks = np.random.normal(size = t)
     
    assert(len(y0) == len(phi))
    
    yList = y0
    
    binom = np.array([sp.gamma(j-dcoeff) / (sp.gamma(j+1) * sp.gamma(-dcoeff)) for j in range(0, t)])  
    binom[np.isnan(binom)] = 0.
    
    for i in range(len(psi), t):
        y = 0
        # MA
        
        for q in range(1, len(psi) + 1):
            
            y += psi[q-1] * shocks[i-q]
            
        for d in range(1, i + 1):
            
            y += -binom[d] * yList[i-d] 
        # AR
        for p in range(1, len(phi) + 1):
            
            y += phi[p-1] * yList[i-p]
            
        y += shocks[i]

        yList.append(y)
        
    return np.array(yList)

















def arimapdqGaussian(t, phi = [0.1], psi = [0.1], dcoeff = 0):
     
    y0 = list(np.zeros(max(len(phi), len(psi))))
    
    shocks = np.random.normal(size = t)
     
    assert(len(y0) == len(phi))
    
    yList = y0
    
    binom = [(-1)**d * sp.binom(dcoeff, d) for d in range(0, dcoeff+1)]
    
    for i in range(len(psi), t):
        y = 0
        # MA
        
        for q in range(1, len(psi) + 1):
            
            y += psi[q-1] * shocks[i-q]
            
        for d in range(1, len(binom)):
            
            y += -binom[d] * yList[i-d] 
        # AR
        for p in range(1, len(phi) + 1):
            
            y += phi[p-1] * yList[i-p]
            
        y += shocks[i]

        yList.append(y)
        
    return np.array(yList)






def armapqGaussian(t, phi = None, psi = None, y0 = None):
     
    shocks = np.random.normal(size = t)
     
#     assert(len(y0) == len(phi))
    if y0 is None:
        y0 = list(np.zeros(max(len(phi), len(psi))))
    
    yList = y0
    
    if phi is None:
        phi = [0.]
    if psi is None:
        psi = [0.]
    
    for i in range(max(len(psi),len(phi)), t):
        y = 0
        # MA
        for q in range(1, len(psi) + 1):
            y += psi[q-1] * shocks[i-q]
        # AR
        for p in range(1, len(phi) + 1):
            y += phi[p-1] * yList[i-p]
            
        y += shocks[i]

        yList.append(y)
        
    return np.array(yList)



def armapqStudent(t, phi = [0.1], psi = [0.1], df = 10):
     
    y0 = list(np.zeros(max(len(phi), len(psi))))
    
    shocks = np.random.standard_t(df, size = t)
     
    assert(len(y0) == len(phi))
    
    yList = []
    for i in range(max(len(psi),len(phi)), t):
        y = 0
        # MA
        for q in range(1, len(psi) + 1):
            y += psi[q-1] * shocks[i-q]
        # AR
        for p in range(1, len(phi) + 1):
            y += phi[p-1] * yList[i-p]
            
        y += shocks[i]

        yList.append(y)
        
    return np.array(yList)



def arma11Gaussian(t, phi = 0.1, psi = 0.1):
    
    y0 = 0
    
    shocks = np.random.normal(size = t)
    
    yList = [y0]
    for i in range(1,t):
        y = phi * yList[i-1] + psi * shocks[i-1] + shocks[i]
        yList.append(y)
        
    return np.array(yList)


def arma11Student(t, phi = 0.1, psi = 0.1, df = 10):
    
    y0 = 0
    shocks = np.random.standard_t(df, size = t)
    
    yList = [y0]
    for i in range(1,t):
        y = phi * yList[i-1] + psi * shocks[i-1] + shocks[i]
        yList.append(y)
        
    return np.array(yList)



def arpGaussian(t, phi = [0.1], y0 = None):

    if type(phi)!= list:
        raise Exception("phi must be a list")    
    if y0 is None:
        y0 = list(np.zeros(len(phi)))
    
    yList = y0
    
#     assert(len(phi) == len(y0))
    
    for i in range(len(phi), t):
        y = 0
        for p in range(1, len(phi) + 1):
            y += phi[p-1] * yList[i-p]
        y += np.random.normal()
        
        yList.append(y)

    
    return np.array(yList) 




def maqGaussian(t, psi = [0.1], y0 = None):


    if y0 is None:
        y0 = list(np.zeros(len(psi)))
             
    shocks = np.random.normal(size = t)
     
    yList = []
    for i in range(len(psi), t):
        y = 0
        for n in range(1, len(psi) + 1):
            y += psi[n-1] * shocks[i-n] 
        y += shocks[i]

        yList.append(y)
        
    return np.array(yList)

def maqStudent(t, psi = [0.1], df = 10):
     
    shocks = np.random.standard_t(df, size = t)
     
    yList = []
    for i in range(len(psi), t):
        y = 0
        for n in range(1, len(psi) + 1):
            y += psi[n-1] * shocks[i-n] 
        y += shocks[i]

        yList.append(y)
        
    return np.array(yList)


def ma1Gaussian(t, psi = 0.1):

    shocks = np.random.normal(size = t)
    
    yList = []
    for i in range(1, t):
        y = psi * shocks[i-1] + shocks[i]
        yList.append(y)
        
    return np.array(yList)


def ma1Student(t, psi = 0.1, df = 10):

    shocks = np.random.standard_t(df, size = t)
    
    yList = []
    for i in range(1, t):
        y = psi * shocks[i-1] + shocks[i]
        yList.append(y)
        
    return np.array(yList)
     


def ar1Gaussian(t, phi = 0.1):

    f = lambda phi, y: phi * y + np.random.normal()
    
    y0 = 0.
    
    yList = [y0]
    for i in range(1, t):
        
        tmp = f(phi, yList[i-1])
        yList.append(tmp)
        
    return np.array(yList)#, np.array(VaR), np.array(unVaR)


def arpStudent(t, phi = [0.1],  df = 10, y0 = None):
    
    
    if y0 is None:
        y0 = list(np.zeros(len(phi)))
        
    yList = y0
    
#     assert(len(phi) == len(y0))
    for i in range(len(phi), t):
        y = 0
        for n in range(1, len(phi) + 1):
            y += phi[n-1] * yList[i-n]
        y += np.random.standard_t(df)
        
        yList.append(y)

    
    return np.array(yList) 


def ar1Student(t, phi = 0.1, df = 10):

    y0 = 0
    
    f = lambda phi, y: phi * y + np.random.standard_t(df)
    
    yList = [y0]
    for i in range(1, t):
        
        tmp = f(phi, yList[i-1])
        yList.append(tmp)
        
    return np.array(yList)#, np.array(VaR), np.array(unVaR)



    

