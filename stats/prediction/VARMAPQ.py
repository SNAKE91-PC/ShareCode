"""
Created on Nov 16, 2024

@author: snake91
"""


import numpy as np
import scipy.special as sp
from copy import deepcopy
from stats.mle.simulate import varmapqGaussian

flatten = lambda l: [item for sublist in l for item in sublist]


def fitvarmapq(data, pMatrix, qMatrix):
    """
        data
            observed values of the process ->
        pMatrix
            AR matrix of coefficients ->
        qMatrix
            MA matrix of coefficients ->
    """

    shocks = np.zeros(data.shape)

    # predict next value

    fitted_values = [varmapqGaussian(1, pMatrix = pMatrix, qMatrix = qMatrix, shocks = shocks[:,i], y0 = data[:,i])[:,1]
                        for i in range(data.shape[1])]

    fitted_values = flatten(fitted_values)
    fitted_values = np.array(fitted_values).reshape(data.shape)

    return fitted_values
