'''
Created on 2 Jan 2020

@author: snake91
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
from datetime import datetime
import os
import pandas as pd
import sqlalchemy as sql

import time
# from ptfmgt.optim import MeanVaREmpiricalOptim


import pandas_datareader.data as web
# from pandas_datareader._utils import RemoteDataError

conn_str = (
    "DRIVER={PostgreSQL Unicode};" 
    "DATABASE=postgres;"
    "UID=Snake91;"
    "PWD=QWEiop123@;"
    "SERVER=localhost;"
    "PORT=5432;"
    )


conn_str = 'postgresql://postgres:postgres@localhost:5432/postgres'
db = sql.create_engine(conn_str)

# conn = pyodbc.connect(conn_str)


sym = ['ABT', 'ADBE', 'ADT', 'AAP', 
       'AES', 'AET', 'AFL', 'AMG', 'A', 'GAS', 'APD', 'ARG', 
       'AKAM', 'AA', 'AGN', 'ALXN', 'ALLE', 'ADS', 'ALL', 
       'ALTR', 'MO', 'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 
       'AIG', 'AMT', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 
       'APC', 'ADI', 'AON', 'APA', 'AIV', 'AMAT', 'ADM', 
       'AIZ', 'T', 'ADSK', 'ADP', 'AN', 'AZO', 'AVGO', 'AVB', 
       'AVY', 'BHI', 'BLL', 'BAC', 'BK', 'BCR', 'BXLT', 'BAX', 
       'BBT', 'BDX', 'BBBY', 'BRK-B', 'BBY', 'BLX', 'HRB', 'BA', 
       'BWA', 'BXP', 'BSK', 'BMY', 'BRCM', 'BF-B', 'CHRW', 'CA', 
       'CVC', 'COG', 'CAM', 'CPB', 'COF', 'CAH', 'HSIC', 'KMX', 
       'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNP', 'CTL', 'CERN', 
       'CF', 'SCHW', 'CHK', 'CVX', 'CMG', 'CB', 'CI', 'XEC', 'CINF', 
       'CTAS', 'CSCO', 'C', 'CTXS', 'CLX', 'CME', 'CMS', 'COH', 'KO',
        'CCE', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CSC', 'CAG', 'COP', 'CNX', 
        'ED', 'STZ', 'GLW', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 
        'DHR', 'DRI', 'DVA', 'DE', 'DLPH', 'DAL', 'XRAY', 'DVN', 'DO', 
        'DTV', 'DFS', 'DISCA', 'DISCK', 'DG', 'DLTR', 'D', 'DOV', 'DOW', 
        'DPS', 'DTE', 'DD', 'DUK', 'DNB', 'ETFC', 'EMN', 'ETN', 'EBAY', 
        'ECL', 'EIX', 'EW', 'EA', 'EMC', 'EMR', 'ENDP', 'ESV', 'ETR', 'EOG', 
        'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ES', 'EXC', 'EXPE', 'EXPD', 
        'ESRX', 'XOM', 'FFIV', 'FB', 'FAST', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 
        'FSIV', 'FLIR', 'FLS', 'FLR', 'FMC', 'FTI', 'F', 'FOSL', 'BEN', 'FCX', 'FTR', 
        'GME', 'GPS', 'GRMN', 'GD', 'GE', 'GGP', 'GIS', 'GM', 'GPC', 'GNW', 'GILD', 'GS', 
        'GT', 'GOOGL', 'GOOG', 'GWW', 'HAL', 'HBI', 'HOG', 'HAR', 'HRS', 'HIG', 'HAS',
         'HCA', 'HCP', 'HCN', 'HP', 'HES', 'HPQ', 'HD', 'HON', 'HRL', 'HSP', 'HST', 
         'HCBK', 'HUM', 'HBAN', 'ITW', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 
         'INTU', 'ISRG', 'IVZ', 'IRM', 'JEC', 'JBHT', 'JNJ', 'JCI', 'JOY', 'JPM', 
         'JNPR', 'KSU', 'K', 'KEY', 'GMCR', 'KMB', 'KIM', 'KMI', 'KLAC', 'KSS', 'KRFT', 'KR', 'LB', 'LLL', 'LH', 'LRCX', 'LM', 'LEG', 'LEN', 'LVLT', 'LUK', 'LLY', 'LNC', 'LLTC', 'LMT', 'L', 'LOW', 'LYB', 'MTB', 'MAC', 'M', 'MNK', 'MRO', 'MPC', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MAT', 'MKC', 'MCD', 'MHFI', 'MCK', 'MJN', 'MMV', 'MDT', 'MRK', 'MET', 'KORS', 'MCHP', 'MU', 'MSFT', 'MHK', 'TAP', 'MDLZ', 'MON', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MUR', 'MYL', 'NDAQ', 'NOV', 'NAVI', 'NTAP', 'NFLX', 'NWL', 'NFX', 'NEM', 'NWSA', 'NEE', 'NLSN', 'NKE', 'NI', 'NE', 'NBL', 'JWN', 'NSC', 'NTRS', 'NOC', 'NRG', 'NUE', 'NVDA', 'ORLY', 'OXY', 'OMC', 'OKE', 'ORCL', 'OI', 'PCAR', 'PLL', 'PH', 'PDCO', 'PAYX', 'PNR', 'PBCT', 'POM', 'PEP', 'PKI', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PBI', 'PCL', 'PNC', 'RL', 'PPG', 'PPL', 'PX', 'PCP', 'PCLN', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RRC', 'RTN', 'O', 'RHT', 'REGN', 'RF', 'RSG', 'RAI', 'RHI', 'ROK', 'COL', 'ROP', 'ROST', 'RLC', 'R', 'CRM', 'SNDK', 'SCG', 'SLB', 'SNI', 'STX', 'SEE', 'SRE', 'SHW', 'SIAL', 'SPG', 'SWKS', 'SLG', 'SJM', 'SNA', 'SO', 'LUV', 'SWN', 'SE', 'STJ', 'SWK', 'SPLS', 'SBUX', 'HOT', 'STT', 'SRCL', 'SYK', 'STI', 'SYMC', 'SYY', 'TROW', 'TGT', 'TEL', 'TE', 'TGNA', 'THC', 'TDC', 'TSO', 'TXN', 'TXT', 'HSY', 'TRV', 'TMO', 'TIF', 'TWX', 'TWC', 'TJK', 'TMK', 'TSS', 'TSCO', 'RIG', 'TRIP', 'FOXA', 'TSN', 'TYC', 'UA', 'UNP', 'UNH', 'UPS', 'URI', 'UTX', 'UHS', 'UNM', 'URBN', 'VFC', 'VLO', 'VAR', 'VTR', 'VRSN', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO', 'VMC', 'WMT', 'WBA', 'DIS', 'WM', 'WAT', 'ANTM', 'WFC', 'WDC', 'WU', 'WY', 'WHR', 'WFM', 'WMB', 'WEC', 'WYN', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XL', 'XYL', 'YHOO', 'YUM', 'ZBH', 'ZION', 'ZTS']


startDate = datetime(2017, 1, 3)
endDate = datetime(2017, 12, 29)

qryCheck = open("/home/snake91/git/ShareCode/stats/ptfmgt/queries/check.sql").read()

for i in range(len(sym)): 
    
    try:
        db.execute(qryCheck.format(sym[i]))
    except:
        print("tbl_EquityData does not exist yet. Creating at next step")

    res = db.execute(qryCheck.format(sym[i]))
    symstartDate, symendDate = res.fetchall()[0]
    
    if symstartDate is not None and symendDate is not None:
        
        symstartDate = datetime.strptime(symstartDate, '%Y-%m-%d')
        symendDate = datetime.strptime(symendDate, '%Y-%m-%d')
    
        if startDate >= symstartDate and endDate <= symendDate:
            continue
        elif startDate >= symstartDate and endDate >= symendDate:
            startDate = symendDate
        elif startDate <= symstartDate and endDate <= symendDate:
            endDate = symstartDate
    
    print(sym[i], startDate, endDate)
    
    while True:
        try:
            f = web.DataReader(sym[i], "av-daily-adjusted", start=startDate,
                                end=endDate,
                               api_key="IRV9B5XALMD8YWJ2")
            err = None
            break
        except ValueError as err1:
            err = err1
            print(err1, "\n")
            
            break
        except RemoteDataError as err2:
            
            err = err2
            print("Too many requests...sleeping \n")
            
            time.sleep(300)
            
            continue
        else:
            break
        
    if err is not None:
        continue
    
    f.reset_index(inplace = True)
    f.rename(columns = {'index' : 'AOD'}, inplace = True)
    f.insert(1, 'symbol', sym[i])
#     f['sym'] = 
    f.columns = list(map(lambda x: str.upper(x), f.columns))
                         
    f.to_sql(name = "tbl_EquityData", con = db, schema = 'public', if_exists = 'append', index = False)
    
    print(sym[i], "\n")
    
print("")







