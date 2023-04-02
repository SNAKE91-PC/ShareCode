'''
Created on Jun 20, 2021

@author: Snake91
'''


import pandas as pd
import numpy as np

import sklearn.decomposition as skdecomp
import sklearn.feature_extraction as skfeature
from sklearn.linear_model import LinearRegression



import pandas_datareader as pdr
import fix_yahoo_finance as yf
yf.pdr_override() 

startDate = "2021/01/01"
endDate   = "2022/01/01"
stock = pdr.get_data_yahoo("SPY", startDate, endDate)
stock = stock[["Adj Close"]]
stockreturn = ((stock - stock.shift())/stock).dropna()



dates = list(map(lambda x: x.strftime("%Y/%m/%d"), stock.index))



from wsj.wsjScraper import wsjScraper

    
for date in dates:
    scraper = wsjScraper(usr = "apicellavittorio@hotmail.it", pwd = "QWEiop123@", date = date)


    links = scraper.getLinks()
    
    articlesDay = list(map(lambda x: scraper.scrapping(x), links))
    articlesDay = list(filter(lambda x: x is not None, articlesDay))

    lemmatizedArticlesDay = list(map(lambda x: " ".join(x), articlesDay)) 


    tfidvect = skfeature.text.TfidfVectorizer()
    X = tfidvect.fit_transform(lemmatizedArticlesDay)

    D = LinearRegression()
    
    Dfit = D.fit(tfidvect, stock)#Dfit = D.fit(ser, star, sample_weight = A)
    Dpred = D.predict(tfidvect)
    Dscore = D.score(tfidvect,stock)


# import spacy
# 
# nlp = spacy.load("en_core_web_sm")
# 
# doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
# for token in doc:
#     print(token.text, token.pos_, token.dep_)



# w = open(r"D:\1.txt", "wb")
# 
# w.write(content.prettify().encode('ascii', 'ignore'))
# 
# w.close()




print("")









