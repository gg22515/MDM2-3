import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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

xTrain = np.load("xTrain2Fish.npy")[0:6000]
yTrain = np.load("yTrain2Fish.npy")[0:6000]

print("Data Loaded")

xTest = np.load("xTest2Fish.npy")
yTest = np.load("yTest2Fish.npy")

print("Data Loaded")

xTest.shape

model = Sequential()
model.add(Dense(8, activation = "relu", input_shape = (1,8)))
model.add(Dense(100, activation = "relu"))
model.add(Dense(500, activation = "relu"))
model.add(Dense(500, activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dense(4, activation = "linear"))
model.compile(loss = "mse", optimizer = "adam", metrics = ["mse"])

history = model.fit(xTrain, yTrain,
                    epochs = 10)

print(model.predict(xTest[0]))
print(yTest[0])

#['Unnamed: 0', 'Unnamed: 0.1', 'X1', 'Y1', 'H1', 'X2', 'Y2', 'H2',
# 'VX1', 'VY1', 'VX2', 'VY2', 'UX1', 'UY1', 'UX2', 'UY2']
