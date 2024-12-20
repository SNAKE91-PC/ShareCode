"""
Created on Nov 25, 2018

@author: snake91
"""

import numpy as np
import scipy.stats as st
import scipy.special as sp
import scipy.optimize as opt
from stats.mle import constraint as cons

flatten = lambda l: [item for sublist in l for item in sublist]



def est_markovchain(tseries):

    states = set(tseries)
    transmat = np.zeros(shape = (len(states), len(states)))
    for row in range(1, len(tseries)):
        transmat[tseries[row-1], tseries[row]] += 1 

        
    return transmat / np.sum(transmat, axis = 0) 


def maxMGARCHECCCpqN(X, alpha, beta):
    # TODO: parameter parsing should be unified in some other method
    '''
        CCC constant conditional correlation
        sqrt(D) * R * sqrt(D)
        D diagonal variance matrix
        R constant (unconditional) correlation matrix
    '''
    
    
    def varianceEstimation(params, x, a, b):
    
        N = lambda x, mu, sigma: np.float((1./np.sqrt(np.linalg.det(2*np.pi*sigma))) * np.exp(-0.5 * (x-mu).T * np.linalg.inv(sigma) * (x-mu)))
        
        logN = lambda x, mu, sigma: -0.5 * np.log(np.linalg.det(2*np.pi*sigma)) + ( -0.5 * (x-mu).T * np.linalg.inv(sigma) * (x-mu) )
        
        
        nprocess, tLen = x.shape[0], x.shape[1]
        
    #     sigma = np.asmatrix(np.diag(np.array(np.sqrt([np.var(i) for i in x]))))
        
        alpha0Params = np.asmatrix(params[: nprocess]).T
        alphaParams = params[nprocess : nprocess + nprocess**2 * a]
        betaParams = params[nprocess + nprocess**2 * a : nprocess + nprocess**2 * (a + b)]
#         sigma = params[nprocess + nprocess ** 2 * (a + b): ]
#         sigma = np.asmatrix(sigma).reshape((nprocess, nprocess))
         
        aLag = int( len(alphaParams) / nprocess ** 2 )
        bLag = int( len(betaParams) / nprocess ** 2 )
        
        alphaList = []
        for i in range(nprocess**2, len(alphaParams)+1, nprocess**2):
            alphaMatrix = np.asmatrix(alphaParams[i-nprocess**2: i]).reshape((nprocess, nprocess))
            alphaList.append(alphaMatrix)
        
        betaList = []
        for i in range(nprocess**2, len(betaParams)+1, nprocess**2):
            betaMatrix = np.asmatrix(betaParams[i-nprocess**2: i]).reshape((nprocess, nprocess))
            betaList.append(betaMatrix)
            
        idx = max(len(alphaList), len(betaList))
        
        np.random.seed(1)
        SIGMA = np.zeros(shape = (nprocess, max(len(alphaList), len(betaList))))#list(np.random.normal(size = (nprocess , max(len(phiList),len(psiList))), loc = np.mean(x),scale = np.sqrt(np.var(x)))) #/(1+np.sum(psi))
        SIGMA = np.asmatrix(np.vstack(SIGMA))
        
        eMatrix = np.zeros(shape = (nprocess, tLen))
        eMatrix[ : , 0: idx] = np.random.normal(size = (nprocess, idx))
        eMatrix = np.asmatrix(eMatrix)
        L = []
        
        for t in range(idx, tLen):
            
            sigma2 = np.zeros(shape = (nprocess, 1))
            sigma2 += alpha0Params
            
            for a in range(1, aLag + 1):
                
                sigma2 += alphaList[a-1] * np.asarray(x[:, t-a])**2
                
            for b in range(1, bLag + 1):
                
                sigma2 += betaList[b-1] * np.asarray(SIGMA[:, t-b])**2
    
            
            sigma = np.sqrt(sigma2)
    #         corrMatrix = np.corrcoef(x[:, t] - eMatrix[:, t])
            eMatrix[:, t] = x[:, t] / sigma # filter out garch effects - leave correlated errors
            prob = flatten(logN(x[:, t], np.mean(x), np.diag(sigma2.T.tolist()[0])).tolist())[0]
            
#             e = x[:, t] - prediction
            SIGMA = np.hstack((SIGMA, sigma))
            
            L.append(prob)#np.log(prob))        
    
        L = sum(L)
        
        print(alpha0Params, alphaParams, betaParams, -L, sep = ' ')
        return {'loglikelihood':-L, 'eMatrix' : eMatrix} #'sigma' : SIGMA, 
    
    
    
    nprocess = X.shape[0]
    aLag = alpha#len(alpha)
    bLag = beta#len(beta)
     
#     sigmaBounds = [(10e-10, None) for i in range(0, nprocess ** 2)] #@unusedvariable
#     corrMatrixBounds = [(-1,1) for i in range(nprocess**2)] #@unusedvariable
    bounds = [(10e-10, 0.99), (10e-10, 0.99)] #alpha0
    for lag in range(0, nprocess **2 * (aLag + bLag)): #@unusedvariable
        bounds += [(10e-10, 0.99)] 
     
    bounds = tuple(bounds)# + sigmaBounds)# + corrMatrixBounds)
     
    x0 = [0.1, 0.1] #alpha0 
    for lag in range(0, (aLag + bLag)): #@unusedvariable
        x0 += list(np.diag(np.array([0.1] * nprocess)).flatten())
         
    flatten = lambda l: [item for sublist in l for item in sublist]
#     sigmaX0 = flatten(np.asmatrix(np.diag(np.array(np.sqrt([np.var(i) for i in X])))).tolist())
#     corrMatrixX0 = flatten(np.identity(nprocess).tolist())
#     x0 = tuple(x0 + sigmaX0)# + corrMatrixX0)

    constraints = ({'type': 'ineq',
                    'fun': lambda params:
                            cons.consVARp(params[nprocess :
                                                 nprocess + nprocess**2 * aLag], aLag)},
                   {'type': 'ineq',
                    'fun': lambda params:
                            cons.consVMAq(params[nprocess + nprocess**2 * aLag:
                                                 nprocess + nprocess**2 * (aLag + bLag)], bLag)}
                   )
    paramsX = opt.minimize(lambda params, x, a, b: varianceEstimation(params, x, a, b)['loglikelihood'], 
                                    x0=x0,
                                    args=(X, aLag, bLag),
                                    bounds=bounds,
                                    constraints=constraints
                          )
        
    paramsX = paramsX.x
    alpha0 = np.asmatrix(paramsX[:nprocess]).T
    alphaParams = paramsX[nprocess:nprocess + nprocess**2 * aLag] 
    betaParams  = paramsX[nprocess + nprocess**2 * aLag: nprocess + nprocess**2 * (aLag + bLag)]  
#     sigmaI    = paramsX[nprocess + nprocess**2 * (aLag + bLag) : ]
    
    alphaX = []
    for i in range(nprocess**2, len(alphaParams[: nprocess**2 * aLag])+1, nprocess**2):
        alphaMatrix = np.asmatrix(alphaParams[i-nprocess**2: i]).reshape((nprocess, nprocess))
        alphaX.append(alphaMatrix)
    
    betaX = []
    for i in range(nprocess**2, len(betaParams[: nprocess**2 * bLag])+1, nprocess**2):
        betaMatrix = np.asmatrix(betaParams[i-nprocess**2: i]).reshape((nprocess, nprocess))
        betaX.append(betaMatrix)
            
#     sigmaI = np.reshape(sigmaI, newshape = (nprocess, nprocess))
    
    resDict = varianceEstimation(paramsX, X, alpha, beta)
    
    eMatrix = resDict['eMatrix']

    corrMatrix = np.corrcoef(eMatrix)
    
    covMatrix = np.cov(eMatrix)
    
    return {'alpha0': alpha0, 'alphaX': alphaX, 'betaX': betaX, 'corrMatrix': corrMatrix, 'covMatrix': covMatrix} #sigmaI




def maxGARCHpqN(params, x, a, b):
#     print(params)
    
    N = lambda x, mu, sigma : (1./np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2 / (2*sigma**2))

    alpha = params[ : a + 1]
    beta  = params[a + 1 : ]    
    
    a0 = alpha[0]
    alpha = alpha[1:]
    
    L = []
    sigmaList = [np.sqrt(np.var(x)) for i in range(len(beta))]
    
    idx = max(len(alpha), len(beta))
    
    for i in range(idx, len(x)):
        
        sigma2 = a0
        
        for a in range(1, len(alpha) + 1):
            
            sigma2 += alpha[a-1] * x[i-a]**2   
        
        for b in range(1, len(beta)  + 1):
            
            sigma2 += beta[b-1] * sigmaList[i-b]**2
        
        sigma = np.sqrt(sigma2)
        
        sigmaList.append(sigma)
        
        prob = N(x[i], np.mean(x), sigma)
    
        L.append(np.log(prob))
        
    L = sum(L)
    
    print(-L)
    
    return -L





def maxARCHpN(params, x):

    N = lambda x, mu, sigma : (1./np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2 / (2*sigma**2))

    alpha = params
        
    a0 = alpha[0]
    alpha = alpha[1:]
    
    L = []
    for i in range(len(alpha), len(x)):
        
        sigma2 = a0
        
        for a in range(1, len(alpha) + 1):
            
            sigma2 += alpha[a-1] * x[i-a]**2   
        
        sigma = np.sqrt(sigma2)
        
        prob = N(x[i], np.mean(x), sigma)
    
        L.append(np.log(prob))
        
    L = sum(L)
    
#     print(-L)
    
    return -L



def maxVARMApqN(X, p, q, start_guess_p = None, start_guess_q = None):
    """
        X
            data of the process -> (obs, nprocess)
        p
            AR order of the process -> int
        q
            MA order of the process -> int
        start_guess_p
            starting guess (one per process) -> list[int]
        start_guess_q
            starting guess (one per process) -> list[int]
    """

    if start_guess_p is None:
        start_guess_p = [0.1 for i in range(p)]

    if start_guess_q is None:
        start_guess_q = [0.1 for i in range(q)]


    def meanEstimation(params, x, p, q, start_guess_p = 0, start_guess_q = 0):
        
        N = lambda x, mu, sigma: np.float((1./np.sqrt(np.linalg.det(2*np.pi*sigma)))
                                          * np.exp(-0.5 * (x-mu).T * np.linalg.inv(sigma) * (x-mu)))

        logN = lambda x, mu, sigma: -0.5 * np.log(np.linalg.det(2*np.pi*sigma)) + ( -0.5 * (x-mu).T * np.linalg.inv(sigma) * (x-mu) )
        
        nprocess, tLen = x.shape[0], x.shape[1]
    #     sigma = np.asmatrix(np.diag(np.array(np.sqrt([np.var(i) for i in x]))))
        
        phiParams = params[ : nprocess**2 * p]
        psiParams = params[nprocess**2 * p : nprocess**2 * (p + q)]
        sigma = params[nprocess ** 2 * (p + q): ]
        sigma = np.asmatrix(sigma).reshape((nprocess, nprocess))
         
        pLag = int( len(phiParams) / nprocess ** 2 )
        qLag = int( len(psiParams) / nprocess ** 2 )
        
        phiList = []
        for i in range(nprocess**2, len(phiParams)+1, nprocess**2):
            phiMatrix = np.asmatrix(phiParams[i-nprocess**2: i]).reshape((nprocess, nprocess))
            phiList.append(phiMatrix)
        
        psiList = []
        for i in range(nprocess**2, len(psiParams)+1, nprocess**2):
            psiMatrix = np.asmatrix(psiParams[i-nprocess**2: i]).reshape((nprocess, nprocess))
            psiList.append(psiMatrix)
            
        np.random.seed(1)
        eList = np.zeros(shape = (nprocess, max(len(phiList), len(psiList))))#list(np.random.normal(size = (nprocess , max(len(phiList),len(psiList))), loc = np.mean(x),scale = np.sqrt(np.var(x)))) #/(1+np.sum(psi))
        eList = np.asmatrix(np.vstack(eList))
        
    #     for i in range(0, len(phiList)):
    #         print 'phi', i+1
    #         print phiList[i]
    #      
    #     for i in range(0, len(psiList)):
    #         print 'psi', i+1
    #         print psiList[i]
    #          
    #     print 'sigma'
    #     print sigma
    #     print '\n'
    #     print phiList, psiList
        
        L = []
        
        for t in range(max(len(phiList),len(psiList)), tLen):
            
            prediction = np.zeros(shape = (nprocess, 1))
            
            for p in range(1, pLag + 1):
                
                prediction += phiList[p-1] * x[:, t-p]
                
            for q in range(1, qLag + 1):
                
                prediction += psiList[q-1] * eList[:, t-q]
    
            
            prob = logN(x[:, t], prediction, sigma)
            
            e = x[:, t] - prediction
            
            eList = np.hstack((eList, e))
            
            L.append(prob)#np.log(prob))        
    
        L = sum(L)
        
#         print(-L)
        return -L

    nprocess = X.shape[0]
    pLag = p
    qLag = q
    
    sigmaBounds = [(10e-10, None) for i in range(0, nprocess ** 2)]
    bounds =tuple( [(-0.99, 0.99) for i in range(0, nprocess ** 2 * (pLag + qLag))] + sigmaBounds)
    # x0 = tuple([0.] * nprocess**2 * pLag)
    
    x0 = []
    # for lag in range(0, (pLag + qLag)):
    #     x0 += list(np.diag(np.array([0.1] * nprocess)).flatten())

    for lag in range(0, pLag):
        x0 += list(np.diag(np.array([start_guess_p[lag]] * nprocess)).flatten())

    for lag in range(0, qLag):
        x0 += list(np.diag(np.array([start_guess_q[lag]] * nprocess)).flatten())


    flatten = lambda l: [item for sublist in l for item in sublist]
    sigmaX0 = flatten(np.asmatrix(np.diag(np.array(np.sqrt([np.var(i) for i in X])))).tolist())
    x0 = np.array(tuple(x0 + sigmaX0))

    constraints = (
                    {'type': 'ineq',
                    'fun': lambda params: cons.consVARp(params[: nprocess**2 * pLag], pLag)},
                    {'type': 'ineq',
                    'fun': lambda params: cons.consVMAq(params[nprocess**2 * pLag: nprocess**2 * (pLag + qLag)], qLag)}
                   )

    paramsX = opt.minimize(meanEstimation, 
                                x0=x0,
                                args=(X, pLag, qLag),
                                bounds=bounds,
                                constraints=constraints
                      )

    # method = 'L-BFGS-B',
    # tol = 10e-16

    phiX = []
    psiX = []
    
    phiParams = paramsX.x[ : nprocess**2 * pLag]
    psiParams = paramsX.x[nprocess**2 * pLag : nprocess**2 * (pLag + qLag)]
    sigma = paramsX.x[nprocess**2 * (pLag + qLag) : ]
    
#     logL.maxVARMApqN(list(phiParams) + list(psiParams) + list(sigma), X, pLag, qLag)
    # p1 = 
    # q1 = 
    # 
    sigmaTrue =  flatten(np.asmatrix(np.diag(np.array(np.sqrt([np.var(i) for i in X])))).tolist())
#     logL.maxVARMApqN(tuple(flatten(p1.tolist()) + flatten(q1.tolist()) + sigmaTrue), X, pLag, qLag)
    
    for i in range(nprocess**2, len(phiParams[: nprocess**2 * pLag])+1, nprocess**2):
        phiMatrix = np.asarray(phiParams[i-nprocess**2: i]).reshape((nprocess, nprocess))
        phiX.append(phiMatrix)
    
    
    for i in range(nprocess**2, len(psiParams[:nprocess**2 * qLag ])+1, nprocess**2):
        psiMatrix = np.asarray(psiParams[i-nprocess**2: i]).reshape((nprocess, nprocess))
        psiX.append(psiMatrix)

    # shocks = np.zeros(X.shape) # calculate the mean process
    # fitted_values = varmapq(X.shape[1], pMatrix = phiX, qMatrix = psiX, shocks = shocks, y0 = X[:,0])

    return {'phi' :phiX,
            'psi' : psiX,
            'sigma': sigmaTrue,
            'status' : paramsX.message}
            # 'fitted_values' : fitted_values}


def maxVARpN(params, x):
    
    N = lambda x, mu, sigma: np.float((1./np.sqrt(np.linalg.det(2*np.pi*sigma))) * np.exp(-0.5 * (x-mu).T * np.linalg.inv(sigma) * (x-mu)))
    
    nprocess, tLen = x.shape[0], x.shape[1]
    sigma = np.asmatrix(np.diag(np.array(np.sqrt([np.var(i) for i in x]))))
    
    pLag = len(params) / nprocess ** 2
    
    phiList = []
    for i in range(nprocess**2, len(params)+1, nprocess**2):
        phiMatrix = np.asmatrix(params[i-nprocess**2: i]).reshape((nprocess, nprocess))
        phiList.append(phiMatrix)
        
    print(phiList)
    
    L = []
    
    for t in range(len(phiList), tLen):
        
        prediction = np.zeros(shape = (nprocess, 1))
        
        for p in range(1, pLag + 1):
            
            prediction += phiList[p-1] * x[:, t-p]
            
        
        prob = N(x[:, t], prediction, sigma)
        
        L.append(np.log(prob))        

    L = sum(L)
    
    return -L


def maxVAR1T(params, x):#, options = {'df': 'single'}):

    
    nprocess, t = x.shape[0], x.shape[1]
    phi = np.asmatrix(np.array(params).reshape((nprocess, nprocess)))
#     sigma = np.asmatrix(np.diag(np.array(params[-nprocess:])))#.reshape((nprocess, 1))

#     sigma = np.asmatrix(np.diag(np.array(np.sqrt([np.var(i) for i in x]))))
    extraparams = [st.t.fit(i) for i in x]
    sigma = np.asmatrix(np.diag(map(lambda x: x[2], extraparams)))
    
#     if options['df'] == 'single':
    df = st.t.fit(x)[0]
#     if options['df'] == 'multiple': 
#         df = np.asmatrix(map(lambda x: x[0], extraparams))
#         raise Exception("Not implemented")

    print(phi)
#     print sigma
        
    def T(x, mu, sigma, df):
        
        d = len(x)
        Num = sp.gamma(1. * (d+df)/2)
        Denom = ( 
                  sp.gamma(1.*df/2) * \
                  pow(df * np.pi,1.*d/2) * \
                  pow(np.linalg.det(sigma),1./2) * \
                  pow(np.float(1 + (1./df) * np.dot(np.dot((x - mu).T, np.linalg.inv(sigma)), (x - mu))), 1.* (d+df)/2)
                )
        d = 1. * Num / Denom 
        
        return d
        
                            
    L = []
    for i in range(1, t):
        prob = T(x[:, i], phi * x[:, i-1], sigma, df = df)
        
        L.append(np.log(prob))
        
    L = sum(L)
    
    return -L



def maxVAR1N(params, x):

    
    nprocess, t = x.shape[0], x.shape[1]
    phi = np.asmatrix(np.array(params).reshape((nprocess, nprocess)))
#     sigma = np.asmatrix(np.diag(np.array(params[-nprocess:])))#.reshape((nprocess, 1))

    sigma = np.asmatrix(np.diag(np.array(np.sqrt([np.var(i) for i in x]))))

    print(phi)
#     print sigma
        
    N = lambda x, mu, sigma: np.float((1./np.sqrt(np.linalg.det(2*np.pi*sigma))) * np.exp(-0.5 * (x-mu).T * np.linalg.inv(sigma) * (x-mu)))
                            
    L = []
    for i in range(1, t):
        prob = N(x[:, i], phi * x[:, i-1], sigma)
        
        L.append(np.log(prob))
        
    L = sum(L)
    
    return -L
    


def maxARMApqT(params, x, p, q):
    
    T = lambda x, mu, sigma, df: st.t.pdf(x, df, loc = mu, scale = sigma)
    
#     print p,q    
    phi = params[:p]
    psi = params[p:]
    
#     print phi, psi
#     sigma = params[-2]
#     df = params[-1]

    extraparams = st.t.fit(x)
    df = extraparams[2]
    sigma = extraparams[0]
    
    L = []
    
    np.random.seed(1)
    eList = list(np.random.normal(size = max(len(phi),len(psi)), loc = np.mean(x),scale = np.sqrt(np.var(x)))) #/(1+np.sum(psi))
        
    for t in range(max(len(phi),len(psi)), len(x)):
        
        prediction = 0.
        
        for p in range(1, len(phi) + 1):
            prediction += phi[p-1] * x[t-p]
            
        for q in range(1, len(psi) + 1):
            prediction += psi[q-1] * eList[t-q]
            
        e = x[t] - prediction 
        eList.append(e)
    
        p = T(x[t], prediction, sigma, df)
        L.append(np.log(p))
        
    L = sum(L)

#     print params, -L
    
    return -L










def maxARMApqN(params, x, p, q):
    
    N = lambda x, mu, sigma : (1./np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2 / (2*sigma**2))
    
#     print p,q    
    phi = params[:p]
    psi = params[p:]
    
#     print phi, psi
    sigma = np.sqrt(np.var(x))#params[-1]
    
    L = []
    
    np.random.seed(1)
    mean = np.mean(x[~np.isnan(x)])
    var = np.nanvar(x)
    eList = list(np.random.normal(size = max(len(phi),len(psi)), loc = mean,scale = np.sqrt(var))) #/(1+np.sum(psi))
        
    for t in range(max(len(phi),len(psi)), len(x)):
        
        prediction = 0.
        
        for p in range(1, len(phi) + 1):
            if np.isnan(x[t-p]):
                pass
            else:
                prediction += phi[p-1] * x[t-p]
            
        for q in range(1, len(psi) + 1):
            if np.isnan(x[t-p]):
                pass
            else:
                prediction += psi[q-1] * eList[t-q]
    
        curr = x[t]
        if np.isnan(x[t]):
            curr = mean
        if np.isnan(prediction):
            prediction = mean

        e = curr - prediction 
        eList.append(e)
        
        p = N(curr, prediction, sigma)
        L.append(np.log(p))
        
    L = sum(L)

#     print params, -L
    
    return -L


def maxARMA11T(params, x):
    
    T = lambda x, mu, sigma, df: st.t.pdf(x, df, loc = mu, scale = sigma)
    
    phi = params[0]
    psi = params[1]
#     sigma = params[2]
#     df = params[3]
    
    extraparams = st.t.fit(x)
    sigma = extraparams[2]
    df = extraparams[0]
    
    L = []
    
    e0 = 0
    eList = [e0]
    
    for t in range(1, len(x)):
        
        e = x[t] - psi * eList[t-1] - phi * x[t-1] 
        eList.append(e)
    
        p = T(x[t], psi * eList[t-1] + phi * x[t-1], sigma, df)
        L.append(np.log(p))
        
    L = sum(L)

    print(params) #-L
    
    return -L



def maxARMA11N(params, x):
    
    N = lambda x, mu, sigma : (1./np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2 / (2*sigma**2))
    
    phi = params[0]
    psi = params[1]
#     sigma = params[2]
    
    sigma = np.sqrt(np.var(x))
    L = []
    
    e0 = 0
    eList = [e0]
    
    for t in range(1, len(x)):
        
        e = x[t] - psi * eList[t-1] - phi * x[t-1] 
        eList.append(e)
    
        p = N(x[t], psi * eList[t-1] + phi * x[t-1], sigma)
        L.append(np.log(p))
        
    L = sum(L)

    print(params)# -L
    
    return -L
    
    
def maxAR1N(params, x):
        
    N = lambda x, mu, sigma : (1./np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2 / (2*sigma**2))
    
    phi = params#[0]
#     sigma = params[1]
    sigma = np.sqrt(np.var(x))
    L = []
    
    for t in range(1, len(x)):
        p = N(x[t],  phi * x[t-1], sigma)
        L.append(np.log(p)) 
        
    L = sum(L)
    
#     print(params)#, -L
    return -L


def maxARpN(params, x, estimation = 'MAXML', sigma = None):


    N = lambda x, mu, sigma : (1./np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2 / (2*sigma**2))
    
    phi = params#[0:-1]
#     sigma = params[-1]
    
    if sigma is None:
        sigma = np.sqrt(np.var(x)) #
    else:
        sigma = sigma
        
    L = []
    
    for t in range(len(phi), len(x)):

        prediction = 0

        for p in range(1, len(phi) + 1):
        
            prediction += phi[p-1] * x[t-p]
            
        prob = N(x[t], prediction, sigma)
        
        L.append(np.log(prob))
        
    if estimation == 'EM':
        pass
    elif estimation == 'MAXML':
        L = -sum(L)
    else:
        raise Exception("invalid estimation parameter")

    return L
    



def maxAR1T( params, x ):
    
    T = lambda x, mu, sigma, df: st.t.pdf(x, df, loc = mu, scale = sigma)
    
    phi = params#[0] 
#     sigma = params[1]
#     df = params[2]

    extraparams = st.t.fit(x)
    sigma = extraparams[2]
    df = extraparams[0]
    L = []
    
    for t in range(1, len(x)):
        p = T(x[t], phi * x[t-1], sigma, df = df)
        L.append(np.log(p))
        
    L = sum(L)
    
    print(params)#, -L
    
    return -L


def maxARpT(params, x):
    
    T = lambda x, mu, sigma, df: st.t.pdf(x, df, loc = mu, scale = sigma)
    
    phi = params#[0:-2]
#     sigma = params[-2]
#     df = params[-1]

    extraparams = st.t.fit(x)
    sigma = extraparams[2]
    df = extraparams[0]
    
    L = []
    
    for t in range(len(phi), len(x)):

        prediction = 0

        for p in range(1, len(phi) + 1):
        
            prediction += phi[p-1] * x[t-p]
        
        prob = T(x[t], prediction, sigma, df = df)
        
        L.append(np.log(prob))
        
    L = sum(L)

    return -L



def maxMAqN( params, x ):
    
    N = lambda x, mu, sigma : (1./np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2 / (2*sigma**2))
    
    psi = params#[0:-1]
#     sigma = params[-1]
    sigma = np.sqrt(np.var(x))

    L = []
    np.random.seed(1)
    eList = list(np.random.normal(size = len(psi), loc = np.mean(x),scale = np.sqrt(np.var(x)))) #/(1+np.sum(psi))
#     eList = [0. for i in range(len(psi))] #@Unusedvariable
    for t in range(len(psi), len(x)):
        
        prediction = 0
        
        for q in range(1, len(psi) + 1):
            
            prediction += psi[q-1] * eList[t-q]
                    
        p = N(x[t], prediction, sigma)
        
        e = x[t] - prediction 
        eList.append(e)
#         
        L.append(np.log(p))
        
    L = sum(L)

    return -L



def maxMA1N( params, x ):
    
    N = lambda x, mu, sigma : (1./np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2 / (2*sigma**2))
    
    psi = params#[0]
#     sigma = params[1]

    sigma = np.sqrt(np.var(x))
    L = []
    
    e0 = 0
    eList = [e0]
    for t in range(1, len(x)):
        
#         e = x[t] - psi * eList[t-1] 
#         eList.append(e)
        
        p = N(x[t], psi * eList[t-1], sigma)
        e = x[t] - psi * eList[t-1] 
        eList.append(e)
#         
        L.append(np.log(p))
        
    L = sum(L)

#     print params, -L
    
    return -L


def maxMA1T( params, x):
    
    T = lambda x, mu, sigma, df: st.t.pdf(x, df, loc = mu, scale = sigma)
    
    psi = params#[0]
#     sigma = params[1]
#     df = params[2]

    extraparams = st.t.fit(x)
    sigma = extraparams[2]
    df = extraparams[0]
    
    L = []
    
    e0 = 0
    eList = [e0]
    for t in range(1, len(x)):
        
        e = x[t] - psi * eList[t-1] 
        eList.append(e)
    
        p = T(x[t], psi * eList[t-1], sigma, df)
        L.append(np.log(p))
        
    L = sum(L)

#     print params, -L
    
    return -L
