'''
Created on Feb 8, 2019

@author: snake91
'''


import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt

# x = [2,4,15,20]
# y = [1,2,3,4]
# z = [0, 0, 1, 1]

x = np.random.normal(size = 100, scale = 0.2)
y = np.random.normal(size = 100, scale = 0.2)
a = np.random.normal(size = 100, scale = 0.2)
z = 0.1 * x + 0.2 * y + a

rhoxy = st.pearsonr(x, y)[0]
rhozx = st.pearsonr(z, x)[0]
rhozy = st.pearsonr(z, y)[0]

rhozy_x = ( rhozy - rhozx * rhoxy ) / \
            (np.sqrt(1-rhozx**2) * np.sqrt(1-rhoxy**2)) 
             
rhozx_y = ( rhozx - rhozx * rhoxy ) / \
            (np.sqrt(1-rhozy**2) * np.sqrt(1-rhoxy**2))

u = OLS(z, x).fit()
v = OLS(y, x).fit()

st.pearsonr(u.resid,v.resid)
st.pearsonr(z,y)

u = OLS(z, y).fit()
v = OLS(x, y).fit()

st.pearsonr(u.resid, v.resid)
st.pearsonr(z, x)