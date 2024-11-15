

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Define ARMA models
arma1 = sm.tsa.ArmaProcess([1, -0.5], [1, 0.4])
arma2 = sm.tsa.ArmaProcess([1, -0.3], [1, 0.6])

# Simulation parameters
n = 1000  # Number of observations
switch_prob = 0.1  # Probability of switching

# Initialize process
process = np.zeros(n)
current_model = arma1

# Simulate process
for t in range(1, n):
    if np.random.rand() < switch_prob:
        current_model = arma2 if current_model == arma1 else arma1
    process[t] = current_model.generate_sample(nsample=1)

# Convert to pandas DataFrame for better visualization
df = pd.DataFrame(process, columns=['Value'])

# Plot the process
df.plot(title='Simulated Process with Two ARMA Models')
