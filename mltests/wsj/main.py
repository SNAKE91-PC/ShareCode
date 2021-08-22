'''
Created on Jun 20, 2021

@author: Snake91
'''


from bs4 import BeautifulSoup
import requests

from selenium import webdriver

usr = "apicellavittorio@hotmail.it"
pwd = "QWEiop123@"


HEADERS = ({'User-Agent':
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
            'Accept-Language': 'en-US, en;q=0.5'})



wsjdate = '2021/01/01'
r = requests.get("https://www.wsj.com/news/archive/" + wsjdate, auth = (usr, pwd), headers = HEADERS)

content = BeautifulSoup(r.content, "html.parser")


articles = content.find_all("article")

links = [articles[i].find_all("a", {'class' : ""})[0]["href"] for i in range(len(articles))]

    
fp = webdriver.FirefoxProfile(r"C:\Users\Snake91\AppData\Roaming\Mozilla\Firefox\Profiles\pucff60s.default-release")
driver = webdriver.Firefox(fp)


def scrapping(link):
    
    driver.get(link)
    
    # r = requests.get(links[0], auth = (usr, pwd), headers = HEADERS)
    
    # content = BeautifulSoup(r.content, "html.parser")
    
    content = BeautifulSoup(driver.page_source, "html.parser")
    
    try:
        headline = content.find("h1", {"class" : "wsj-article-headline"}).contents[0].replace("\n", "")
        subheadline = content.find_all("h2", {"itemprop" : "description"})[0].contents[0]
        text = content.find_all("div", {"class" : "article-content"})
        
        
        paragraphs = content.find_all("p")
        paragraphs = [paragraphs[i].get_text() for i in range(len(paragraphs))]
        paragraphs = [paragraphs[i].replace("\n", "") for i in range(len(paragraphs))]
        paragraphs = list(filter(lambda x: x!="" and "http" not in x, paragraphs))
        
        paragraphs = "\n".join(paragraphs)
        
        return paragraphs
    except:
        pass

articles = list(map(lambda x: scrapping(x), links))
articles = list(filter(lambda x: x is not None, articles))

driver.close()

import nltk
# from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
#     words_articles = map(lambda article: word_tokenize(article), articles)
    tokenizer = RegexpTokenizer(r'\w+')
    words_articles = map(lambda article: tokenizer.tokenize(article), articles)
except LookupError:
    nltk.download("punkt")
    
try:
    stop_words = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

filtered_articles = [list(filter(lambda x: x.lower() not in stop_words, word_article)) for word_article in words_articles]

try:
    lemmatizer = WordNetLemmatizer()
except LookupError:
    nltk.download("wordnet")


lemmatized_articles = [list(map(lambda word: lemmatizer.lemmatize(word), article)) for article in filtered_articles]


import sklearn.decomposition as skdecomp
import sklearn.feature_extraction as skfeature

lemmatized_articles = list(map(lambda x: " ".join(x), lemmatized_articles))

tfidvect = skfeature.text.TfidfVectorizer()
X = tfidvect.fit_transform(lemmatized_articles)

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









