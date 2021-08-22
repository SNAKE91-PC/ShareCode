'''
Created on Aug 8, 2021

@author: Snake91
'''


import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pandas_datareader.data as pdr

import yfinance as yf

yf.pdr_override()

from datetime import datetime as dt



app = Flask(__name__)
app.debug = True
app.autoreload = False


# @app.route("/myajax", methods=['GET', 'POST'])
# def mypostajaxreq():
#     
#     print(request.form)
#     if request.method == "POST":
#         name = request.form["name"]
#         return " Hello " + name
#     else:
#         return "Just Hello"                
#             
#             

@app.route("/query", methods = ["GET", "POST"])
def myajaxquery():
     
    if request.method == "POST":
        args = list(request.form) 
    elif request.method == "GET":
        args = list(request.args)
    
    
    data = [pdr.get_data_yahoo(args[i], start="2017-01-01", end="2017-04-30") for i in range(len(args))]
    
    for tickerIdx in range(len(data)):
         
        data[tickerIdx].insert(0, 'symbol', args[tickerIdx]) 
         
         
    data = pd.concat(data).reset_index()
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    data = data[["Date","symbol", "Close"]]
    
    return data.to_json(orient = "records")
#     
      
      
            

@app.route("/")
def index():
    
    return render_template("home.html")


if __name__ == "__main__":
    app.run(host = "0.0.0.0")
    
    
    
    
    
    