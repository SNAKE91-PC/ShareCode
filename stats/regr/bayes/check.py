'''
Created on Sep 15, 2019

@author: snake91
'''



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '/home/snake91/check_bayes/'
fileName = 'data_w_assumedvariance'

fileName = 'data_w_assumedvariance_wrong'
df = pd.read_csv(path + fileName + ".csv")

b0 = df['b0'].get_values()

plt.hist(b0, bins = 500)