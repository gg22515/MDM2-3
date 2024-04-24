import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

import time
LOG_DIR = f"{int(time.time())}"

##print(list(trainData.columns.values))

##df = pd.read_csv("5FishMainData.csv", low_memory = False)
##
##invalid_strings = ['#DIV/0!', 'NA', '#VALUE!']
##for col in df.columns:
##    for invalid_str in invalid_strings:
##        df = df[df[col] != invalid_str]
##
### Convert remaining values to floats
##df = df.apply(pd.to_numeric, errors='ignore')
##
### Drop rows with NaN values
##df = df.dropna()
##
##df.to_csv("5FishMainDataCleaned.csv")

#['Unnamed: 0', 'X1', 'Y1', 'H1', 'X2', 'Y2', 'H2', 'X3', 'Y3', 'H3', 'X4', 'Y4',
# 'H4', 'X5', 'Y5', 'H5', 'VX1', 'VY1', 'UX1', 'UY1', 'VX2', 'VY2', 'UX2', 'UY2',
# 'VX3', 'VY3', 'UX3', 'UY3', 'VX4', 'VY4', 'UX4', 'UY4', 'VX5', 'VY5', 'UX5', 'UY5']

xTrainLoad = np.load("xTrain5Fish.npy")
yTrainLoad = np.load("yTrain5Fish.npy")

p = np.random.permutation(len(xTrainLoad))

xTrain = xTrainLoad[p]
yTrain = yTrainLoad[p]


def build_model(hp):

##model.add(Dense(8, activation = "relu", input_shape = (1,8)))
##model.add(Dense(100, activation = "relu"))
##model.add(Dense(500, activation = "relu"))
##model.add(Dense(500, activation = "relu"))
##model.add(Dense(100, activation = "relu"))
##model.add(Dense(100, activation = "relu"))
##model.add(Dense(4, activation = "linear"))
##model.compile(loss = "mse", optimizer = "adam", metrics = ["mse"])
  
    model = Sequential()

    model.add(Dense(hp.Int('input_units',
        min_value= 18,
        max_value= 72,
        step=18), activation = "relu", input_shape = (1,18)))

    for i in range(hp.Int('n_layers', 1, 8)):
      
      model.add(Dense(hp.Int(f'dense_{i}_units',
        min_value= 10,
        max_value= 500,
        step=50), activation = "relu"))

    model.add(Dense(2, activation = "linear"))

    model.compile(optimizer="adam",
                  loss="mse",
                  metrics=["mse"])
    
    return model

##tuner = RandomSearch(
##    build_model,
##    objective="mse",
##    max_trials=16,  # how many model variations to test?
##    executions_per_trial=1,  # how many trials per variation? (same model could perform differently)
##    directory=LOG_DIR)
##
##tuner.search(x=xTrain,
##             y=yTrain,          
##             epochs=5)
##
##tuner.results_summary()
##tuner.get_best_hyperparameters()[0].values
##

model = Sequential()
model.add(Dense(900, activation = "relu", input_shape = (1,20)))
model.add(Dense(900, activation = "relu"))
model.add(Dense(900, activation = "relu"))
model.add(Dense(900, activation = "relu"))
#model.add(Dropout(rate = 0.15))
model.add(Dense(4, activation = "linear"))
model.compile(loss = "mse", optimizer = "adam", metrics = ["mse"])


#model = load_model("5FishAllModelNew2")

history = model.fit(xTrain, yTrain,
                    epochs = 5,
                    batch_size = 50000
                    )

#model.save("5FishAllModelVel")

#Feed previous location into next iteration for visualisation 

#df = pd.read_csv("5FishMainDataCleaned.csv", low_memory = False)
#testData = df.iloc[:1000,:]


#MSE 184

#[36m |-dense_0_units: 260[0m
#[34m |-dense_1_units: 410[0m
#[36m |-dense_2_units: 10[0m
#[34m |-dense_3_units: 60[0m
#[36m |-dense_4_units: 460[0m
#[34m |-dense_5_units: 360[0m
#[36m |-dense_6_units: 110[0m
#[34m |-dense_7_units: 160[0m
#[36m |-input_units: 72[0m
#
##df = pd.read_csv("5FishMainDataCleaned.csv", low_memory = False)
##trainData = df.iloc[1000:,:]
##testData = df.iloc[:1000,:]
##
##xTrain = []
##yTrain = []
##
##xTest = []
##yTest = []
##
##fish = ["1", "2", "3", "4", "5"]
##
##for fishIndex in range(len(fish)):
##  print(fishIndex)
##  fishCache = fish.copy()
##  currentFish = fishCache.pop(fishIndex)
##  #print(currentFish, fishCache)
##  for i in range(len(trainData.X1) - 2):
##    xTrain.append([
##      trainData["X" + currentFish].iloc[i],
##      trainData["Y" + currentFish].iloc[i],
##      trainData["VX" + currentFish].iloc[i],
##      trainData["VY" + currentFish].iloc[i],
##      trainData["X" + fishCache[0]].iloc[i],
##      trainData["Y" + fishCache[0]].iloc[i],
##      trainData["VX" + fishCache[0]].iloc[i],
##      trainData["VY" + fishCache[0]].iloc[i],
##      trainData["X" + fishCache[1]].iloc[i],
##      trainData["Y" + fishCache[1]].iloc[i],
##      trainData["VX" + fishCache[1]].iloc[i],
##      trainData["VY" + fishCache[1]].iloc[i],
##      trainData["X" + fishCache[2]].iloc[i],
##      trainData["Y" + fishCache[2]].iloc[i],
##      trainData["VX" + fishCache[2]].iloc[i],
##      trainData["VY" + fishCache[2]].iloc[i],
##      trainData["X" + fishCache[3]].iloc[i],
##      trainData["Y" + fishCache[3]].iloc[i],
##      trainData["VX" + fishCache[3]].iloc[i],
##      trainData["VY" + fishCache[3]].iloc[i]
##      ])
##    
##    yTrain.append([
##      trainData["X" + currentFish].iloc[i+1],
##      trainData["Y" + currentFish].iloc[i+1],
##      trainData["VX" + currentFish].iloc[i+1],
##      trainData["VY" + currentFish].iloc[i+1],
##      ])
##
##xTrainFinal = np.array(xTrain).reshape(len(xTrain),1,20)
##yTrainFinal = np.array(yTrain).reshape(len(yTrain),1,4)
##
##np.save("xTrain5Fish.npy", xTrainFinal)
##np.save("yTrain5Fish.npy", yTrainFinal)

#print(list(df.columns.values))

##for i,file in enumerate(os.listdir("2fishappended")):
##  print(i)
##  if i == 0:
##    df = pd.read_csv("2fishappended/" + file, low_memory=False)
##
##  elif i > 0:
##    dfCache = pd.read_csv("2fishappended/" + file, low_memory=False)
##    pd.concat([df,dfCache] ,axis = 0)
##
##df.to_csv("2FishDataset.csv")
##

##df = pd.read_csv("2FishDatasetCleaned.csv", low_memory = False)
##trainData = df.iloc[1000:,:]
##testData = df.iloc[:1000,:]
##
##xTrain = []
##yTrain = []
##
##xTest = []
##yTest = []
##
##for i in range(len(trainData.X1) - 2):
##  xTrain.append([
##    trainData.X1.iloc[i],
##    trainData.Y1.iloc[i],
##    trainData.VX1.iloc[i],
##    trainData.VY1.iloc[i],
##    trainData.X2.iloc[i],
##    trainData.Y2.iloc[i],
##    trainData.VX2.iloc[i],
##    trainData.VY2.iloc[i]
##    ])
##
##  yTrain.append([
##    trainData.X1.iloc[i + 1],
##    trainData.Y1.iloc[i + 1],
##    trainData.VX1.iloc[i + 1],
##    trainData.VY1.iloc[i + 1]
##    ])
##print("e")
##for i in range(len(testData.X1) - 2):
##  xTest.append([
##    testData.X1.iloc[i],
##    testData.Y1.iloc[i],
##    testData.VX1.iloc[i],
##    testData.VY1.iloc[i],
##    testData.X2.iloc[i],
##    testData.Y2.iloc[i],
##    testData.VX2.iloc[i],
##    testData.VY2.iloc[i]
##    ])
##
##  yTest.append([
##    testData.X1.iloc[i + 1],
##    testData.Y1.iloc[i + 1],
##    testData.VX1.iloc[i + 1],
##    testData.VY1.iloc[i + 1]
##    ])
##
##xTrainFinal = np.array(xTrain).reshape(len(xTrain),1,8)
##yTrainFinal = np.array(yTrain).reshape(len(yTrain),1,4)
##
##xTestFinal = np.array(xTest).reshape(len(xTest),1,8)
##yTestFinal = np.array(yTest).reshape(len(yTest),1,4)
##
##np.save("xTrain2Fish.npy", xTrainFinal)
##np.save("yTrain2Fish.npy", yTrainFinal)
##
##np.save("xTest2Fish.npy", xTestFinal)
##np.save("yTest2Fish.npy", yTestFinal)

##xTrain = np.load("xTrain2Fish.npy")[0:6000]
##yTrain = np.load("yTrain2Fish.npy")[0:6000]
##
##print("Data Loaded")
##
##xTest = np.load("xTest2Fish.npy")
##yTest = np.load("yTest2Fish.npy")
##
##print("Data Loaded")
##
##xTest.shape
##
##model = Sequential()
##model.add(Dense(8, activation = "relu", input_shape = (1,8)))
##model.add(Dense(100, activation = "relu"))
##model.add(Dense(500, activation = "relu"))
##model.add(Dense(500, activation = "relu"))
##model.add(Dense(100, activation = "relu"))
##model.add(Dense(100, activation = "relu"))
##model.add(Dense(4, activation = "linear"))
##model.compile(loss = "mse", optimizer = "adam", metrics = ["mse"])
##
##history = model.fit(xTrain, yTrain,
##                    epochs = 10)
##
##print(model.predict(xTest[0]))
##print(yTest[0])
