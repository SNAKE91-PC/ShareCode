'''
Created on Jun 17, 2021

@author: Snake91
'''

# -*- coding: ascii -*-

import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.normal(size = 1000).reshape(100,10))

import subprocess

exe =  r"C:\Users\Snake91\source\repos\ConsoleApplication1\x64\Debug\ConsoleApplication1.exe"

process = subprocess.Popen([exe],
                           stdin =  subprocess.PIPE,
                      stdout = subprocess.PIPE,
                     stderr=subprocess.PIPE, shell = True)

# process = subprocess.Popen(["ping", b"www.google.com"],
#                            stdin =  subprocess.PIPE,
#                       stdout = subprocess.PIPE,
#                      stderr=subprocess.PIPE, shell = True)



stdout, stderr = process.communicate(df.to_records().tobytes())


print("")





