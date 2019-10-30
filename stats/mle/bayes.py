'''
Created on Sep 8, 2019

@author: snake91
'''

import pymc3 as pm
from theano import scan, shared

import numpy as np

from mle.simulate import armapqGaussian

def build_model():
#     y = shared(np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32))
    y = shared(armapqGaussian(t = 500, phi  = [0.5], psi = [0.2]))
    with pm.Model() as arma_model:
        sigma = pm.HalfNormal('sigma', 5.)
        theta = pm.Normal('theta', 0., sigma=1.)
        phi = pm.Normal('phi', 0., sigma=2.)
        mu = pm.Normal('mu', 0., sigma=10.)

        err0 = y[0] - (mu + phi * mu)

        def calc_next(last_y, this_y, err, mu, phi, theta):
            nu_t = mu + phi * last_y + theta * err
            return this_y - nu_t

        err, _ = scan(fn=calc_next,
                      sequences=dict(input=y, taps=[-1, 0]),
                      outputs_info=[err0],
                      non_sequences=[mu, phi, theta])

        pm.Potential('like', pm.Normal.dist(0, sigma=sigma).logp(err))
    return arma_model


def run(n_samples=1000):
    model = build_model()
    with model:
        trace = pm.sample(draws=n_samples,
                          tune=1000,
                          target_accept=.99)

        
    pm.traceplot(trace)

if __name__ == '__main__':
    run()