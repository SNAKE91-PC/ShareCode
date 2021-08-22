'''
Created on 9 Aug 2020

@author: snake91
'''


import pandas as pd
import pandas_datareader as web
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

data = web.get_data_yahoo(["GOOG"])
data.columns = data.columns.droplevel(level = 1)

data["Returns"] = data["Adj Close"] / data["Adj Close"].shift()

lags = 250


arr = np.array([np.array(data["Returns"].shift(i))[lags+1:] for i in range(1, lags + 1)])

norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) 

x_norm_train   = norm_arr[1:, :]#norm_arr.shape[1]]
Y_norm_train = norm_arr[0, :]#norm_arr.shape[1]]
Y_norm_train = np.reshape(Y_norm_train, (1,norm_arr.shape[1]))

Y_train = arr[0, :]
Y_train = np.reshape(Y_train, (1, arr.shape[1]))

# model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[0]]),
# #         tf.keras.layers.Dense(250, activation = "relu", name = "layer1"),
#         tf.keras.layers.Dense(52, activation = "relu", name = "layer2"),
#         tf.keras.layers.Dense(12, activation = "relu", name = "layer3"),
#         tf.keras.layers.Dense(1,  activation = "relu", name = "output") #activation = "relu",
#         ]
#     )

model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[x_norm_train.shape[1], x_norm_train.shape[0]]),
#         tf.keras.layers.Dense(250, activation = "relu", name = "layer1"),
        tf.keras.layers.Dense(52, activation = "relu", name = "layer2"),
        tf.keras.layers.Dense(12, activation = "relu", name = "layer3"),
        tf.keras.layers.Dense(1,  activation = "relu", name = "output") #activation = "relu",
        ]
    )


model.compile(loss="mse",
              optimizer=tf.keras.optimizers.SGD(lr=1e-3),
              metrics=["mean_squared_error"])

res = model.fit(np.asarray(x_norm_train).T, np.asarray(Y_norm_train).T,
        epochs=1000
        )

loss = model.evaluate(np.asarray(x_norm_train).T, np.asarray(Y_norm_train).T)

Y_norm_fit = model.predict(np.asarray(x_norm_train).T)
# model.predict_proba(np.asarray(x_train).T)

# plt.plot(Y_norm_fit, label = "fit")
# plt.plot(Y_train.T, label = "train")
# plt.legend()
# plt.show()


Y_fit = Y_norm_fit * (np.max(arr) - np.min(arr)) + np.min(arr)


# plt.plot(Y_fit, label = "fit")
# plt.plot(Y_train.T, label = "train")
# plt.legend()
# plt.show()

var_norm = np.var((Y_norm_train - Y_norm_fit.T)[0])


scenarios = np.random.normal(scale = np.sqrt(var_norm), size = int(10e4))
dfscenarios = pd.DataFrame(x_norm_train[:,-1]).transpose()
dfscenarios.columns = np.arange(1, len(dfscenarios.columns)+1)

dfscenarios["idx"]  = 1


# scenarios = pd.DataFrame(scenarios, columns = ["Scenario"])    
# scenarios["idx"] = 1


# dfscenarios = pd.merge(df, scenarios, how = "inner", on = "idx")

# del dfscenarios["idx"]

for i in range(1, 500):

    print(i)
    dfscenarios[ 249 + i ] = model.predict( dfscenarios[ list(range(i, 249 + i)) ]    ) 
    dfscenarios[ 249 + i ] = np.percentile(dfscenarios[ 249 + i ].values + scenarios, 99) #["Scenario"]

    scenarios = np.random.normal(scale = np.sqrt(var_norm), size = int(10e4))

    
c = 0


# line99 = dfscenarios[ list(range(1,249+i)) ].apply(lambda x: np.percentile(x,99))
# line1 = dfscenarios[ list(range(1,249+i)) ].apply(lambda x: np.percentile(x,1))
# 
# line50 = dfscenarios[ list(range(1,249+i)) ].apply(lambda x: np.percentile(x,50))



# for line in np.array(dfscenarios.iloc[1:10,: ]):
#     
#     if c==0:
#         
#         plt.plot(line)
#     
#     else:
#         plt.plot(np.hstack([np.array([np.nan] * 249), line[250:]]))
#         
#     c+=1



plt.plot(dfscenarios.values.T)

plt.show()
print("")












