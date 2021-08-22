'''
Created on 27 Jun 2020

@author: snake91
'''




from flask import Flask
from flask import render_template


app = Flask(__name__)


    
    
@app.route("/")
def cv():
    
    return render_template("index.html")


if __name__ == "__main__":
    
    app.run(debug=True)