'''
Created on Aug 18, 2021

@author: Snake91
'''



import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Snake91\Desktop\data.csv", sep = "\t")

vwap = sum(df["Price"] * df["Quantity"] / sum(df["Quantity"]))

df["hour"] = df["Time"].apply(lambda x: x[0:2])

df[["hour", "Price"]].groupby(by = "hour").apply(lambda x: np.average(x) ).sum() /  len(df["hour"].unique())



import scipy.stats as st



def d1(S0, X, r, t, q, sigma):
    
    return ((np.log(S0/X)) + t * (r-q + 0.5*sigma**2)) / (sigma * np.sqrt(t))

def d2(S0, X, r, t, q, sigma):
    
    return d1(S0, X, r, t, q, sigma) - sigma * t


def P(S0, X, r, t, q, sigma):
    
    
    return X * np.exp(-r*t) * st.norm.cdf(-d2(S0, X, r, t, q, sigma)) - S0 * np.exp(-q*t) * st.norm.cdf(-d1(S0, X, r, t, q, sigma))



T = 100

stock = [65.24, 66]
K = [20,20]
ir = [0.01, 0.0120]
t = [T, T+1]
q = [0,0]
vol = [0.39555, 0.38]


tuples = list(zip(stock, K, ir, t, q, vol))




diffmtm = P(*tuples[1]) - P(*tuples[0])


def greek(pos, tuple, h):
    
    P_t0 = P(*tuple)
    
    tuple1 = list(tuple)
    tuple1[pos] = tuple[pos] + h
    
    P_t1 = P(*tuple)
    
    return (P_t1 - P_t0) / h
    
    
    
    
        
shock = 0.01




  
    

    
 #######################################   



print("")