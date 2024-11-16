import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from stats.mle.likelihood import maxVARMApqN
from stats.mle.simulate import varmapqGaussian
from stats.mle.tests.VARIMApdq import pMatrix, qMatrix
from stats.prediction.VARMAPQ import fitvarmapq

# Define the ARMA models
# arma1 = sm.tsa.ArmaProcess([1, -0.3], [1, 0.5])
# arma2 = sm.tsa.ArmaProcess([1, 0.5], [1, 0.3])


# E-step: Calculate the responsibilities
def e_step(data, params1, params2, pi):
    n = len(data)
    resp1 = np.zeros(n)
    resp2 = np.zeros(n)

    for t in range(n):
        likelihood1 = norm.pdf(data[t], loc=params1[0], scale=params1[1])
        likelihood2 = norm.pdf(data[t], loc=params2[0], scale=params2[1])
        resp1[t] = pi * likelihood1
        resp2[t] = (1 - pi) * likelihood2

    resp_sum = resp1 + resp2
    resp1 /= resp_sum
    resp2 /= resp_sum

    return resp1, resp2


# M-step: Update the parameters
def m_step(data, resp1, resp2):
    params1 = np.zeros(2)
    params2 = np.zeros(2)

    params1[0] = np.sum(resp1 * data) / np.sum(resp1)
    params1[1] = np.sqrt(np.sum(resp1 * (data - params1[0]) ** 2) / np.sum(resp1))

    params2[0] = np.sum(resp2 * data) / np.sum(resp2)
    params2[1] = np.sqrt(np.sum(resp2 * (data - params2[0]) ** 2) / np.sum(resp2))

    pi = np.mean(resp1)

    return params1, params2, pi


# Calculate MSE
def calculate_mse(data, fittedvalues):
    residuals = data - fittedvalues
    mse = np.mean(residuals ** 2)
    return mse


# EM algorithm
def em_algorithm(data, max_iter=100, tol=1e-6):
    # Initialize parameters
    params1 = np.array([[0], [0]])  # Initial values for AR, MA
    params2 = np.array([[0], [0]])  # Initial values for AR, MA
    pi = 0.5
    prev_mse1, prev_mse2 = np.inf, np.inf

    for i in range(max_iter):
        # E-step
        resp1, resp2 = e_step(data, params1, params2, pi)

        # M-step
        new_params1, new_params2, new_pi = m_step(data, resp1, resp2)

        # Fit ARMA models
        model1 = maxVARMApqN(np.reshape(data, (1, len(data))), 1, 1,
                             start_guess_p = params1[0],
                             start_guess_q = params1[1]) #sm.tsa.ARIMA(data, order=(1, 0, 1)).fit(start_params=params1)
        model2 = maxVARMApqN(np.reshape(data, (1, len(data))), 1, 1,
                             start_guess_p=params2[0],
                             start_guess_q=params2[1])  # sm.tsa.ARIMA(data, order=(1, 0, 1)).fit(start_params=params1)

        # Update parameters with fitted values
        new_params1 = model1['phi'] + model1['psi']
        new_params2 = model2['phi'] + model2['psi']

        fitted_values_model1 = fitvarmapq(np.reshape(data, (1, len(data))), new_params1[0], new_params1[1])
        fitted_values_model2 = fitvarmapq(np.reshape(data, (1, len(data))), new_params2[0], new_params2[1])

        # Calculate MSE
        mse1 = calculate_mse(data, fitted_values_model1)
        mse2 = calculate_mse(data, fitted_values_model2)

        print(f"Iteration {i + 1}: MSE for ARMA model 1: {mse1}, MSE for ARMA model 2: {mse2}")

        # Check for convergence
        if np.allclose(params1, new_params1, atol=tol) and np.allclose(params2, new_params2, atol=tol) and np.isclose(
                pi, new_pi, atol=tol) and np.isclose(mse1, prev_mse1, atol=tol) and np.isclose(mse2, prev_mse2,
                                                                                               atol=tol):
            break

        params1, params2, pi = new_params1, new_params2, new_pi
        prev_mse1, prev_mse2 = mse1, mse2

    return params1, params2, pi


# Simulate data
n = 1000
data = np.zeros((n))
switch_prob = 0.1

pMatrix1 = [[0.2]]
pMatrix2 = [[0.6]]
qMatrix1 = [[0.2]]
qMatrix2 = [[0.6]]

arma1 = lambda x: varmapqGaussian(1, pMatrix = pMatrix1, qMatrix = qMatrix1, y0 = x)[:,1]
arma2 = lambda x: varmapqGaussian(1, pMatrix = pMatrix2, qMatrix = qMatrix2, y0 = x)[:,1]
current_model = arma1
for t in range(1, n):
    if np.random.rand() < switch_prob:
        current_model = arma2 if current_model == arma1 else arma1
    data[t] = current_model(data[-1])

# Run EM algorithm
params1, params2, pi = em_algorithm(data)
print("Estimated parameters for ARMA model 1:", params1)
print("Estimated parameters for ARMA model 2:", params2)
print("Estimated switching probability:", pi)
