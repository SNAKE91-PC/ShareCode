'''
Created on 9 Aug 2020

@author: snake91
'''


import pandas as pd
import pandas_datareader as web
import tensorflow as tf
import numpy as np


data = web.get_data_yahoo(["GOOG"])
data.columns = data.columns.droplevel(level = 1)

data["Returns"] = data["Adj Close"] / data["Adj Close"].shift()

lags = 50

arr = np.array([np.array(data["Returns"].shift(i))[lags:] for i in range(1, lags + 1)])
arr = np.where(np.isnan(arr), 0, arr)

x_train   = arr[1:, :1100]
Y_train = arr[0, :1100]
Y_train = np.reshape(Y_train, (1,1100))


model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[x_train.shape[0], x_train.shape[1]]),
        tf.keras.layers.Dense(lags, activation = "softmax", name = "layer1"),
        tf.keras.layers.Dense(lags, activation ="softmax", name = "layer2")
        ]
    )

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.SGD(lr=1e-3),
              metrics=["accuracy"])

res = model.fit(np.asarray(x_train).T, np.asarray(Y_train).T,
        epochs=1
        )

print("")

