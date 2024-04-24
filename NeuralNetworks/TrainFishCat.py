import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

xTrainLoad = np.load("xTrainVision.npy")
yTrainLoad = np.load("yTrainVision.npy")

xTestLoad = np.load("xTestVision.npy")
yTestLoad = np.load("yTestVision.npy")

xTest = xTestLoad[0:300]
yTest = yTestLoad[0:300]


p = np.random.permutation(len(xTrainLoad))

xTrain = xTrainLoad[p][0:60000]
yTrain = yTrainLoad[p][0:60000]


xTrainLoadFull = np.load("xTrain5Fish.npy")
yTrainLoadFull = np.load("yTrain5Fish.npy")

xTestLoadFull = np.load("xTest5Fish.npy")
yTestLoadFull = np.load("yTest5Fish.npy")

p = np.random.permutation(len(xTrainLoadFull))

xTrainFull = xTrainLoadFull[p][0:60000]
yTrainFull = yTrainLoadFull[p][0:60000]

xTestFull = xTestLoadFull[0:300]
yTestFull = yTestLoadFull[0:300]


##
model = Sequential()
model.add(Dense(900, activation = "relu", input_shape = (1,20)))
model.add(Dense(900, activation = "relu"))
model.add(Dense(900, activation = "relu"))
model.add(Dense(900, activation = "relu"))
model.add(Dropout(rate = 0.2))
model.add(Dense(4, activation = "linear"))
##model.compile(loss = "mse", optimizer = "rmsprop", metrics = ["mse"])


#model = load_model("5FishVelInfluential")
#opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss = "mse", optimizer = "adam", metrics = ["mse"])

historyFull = model.fit(xTrainFull, yTrainFull,
                    epochs = 5,
                    batch_size = 10000,
                    validation_data = (xTestFull, yTestFull)
                    )

model = Sequential()
model.add(Dense(1000, activation = "relu", input_shape = (1,8)))
model.add(Dense(1000, activation = "relu"))
model.add(Dense(1000, activation = "relu"))
model.add(Dense(1000, activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dropout(rate = 0.2))
model.add(Dense(4, activation = "linear"))

model.compile(loss = "mse", optimizer = "adam", metrics = ["mse"])

historyBias = model.fit(xTrain, yTrain,
                    epochs = 5,
                    batch_size = 10000,
                    validation_data = (xTest, yTest)
                    )

##model = Sequential()
##model.add(Dense(32, activation = "relu", input_shape = (1,8)))
##model.add(Dense(100, activation = "relu"))
##model.add(Dense(100, activation = "relu"))
##model.add(Dense(500, activation = "relu"))
##model.add(Dense(500, activation = "relu"))
##model.add(Dense(100, activation = "relu"))
##model.add(Dense(100, activation = "relu"))
##model.add(Dense(500, activation = "relu"))
##model.add(Dropout(rate = 0.4))
##model.add(Dense(4, activation = "linear"))
##
##model.compile(loss = "mse", optimizer = "adam", metrics = ["mse"])
##
##historyHighDrop = model.fit(xTrain, yTrain,
##                    epochs = 5,
##                    batch_size = 30000,
##                    validation_data = (xTest, yTest)
##                    )
##
##pd.DataFrame(historyNoDrop.history)["loss"].plot(figsize=(8,5), label = "No Dropout")
pd.DataFrame(historyFull.history)["val_mse"].plot(figsize=(8,5), label = "All Seeing Fish")
pd.DataFrame(historyBias.history)["val_mse"].plot(figsize=(8,5), label = "Visual Bias Fish")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.legend()

plt.show()
#model.save("5FishVelInfluential2")

##df = pd.read_csv("training data - five fishCLEANED.csv", low_memory = False)
##
##
##
##
##trainData = df.iloc[1000:len(df)]
##testData = df.iloc[:1000]
##
##xTest = []
##yTest = []
##
##for i in range(len(testData.X1) - 2):
##  if testData["1to2"].iloc[i] == 1:
##    xTest.append([
##      testData.X1.iloc[i],
##      testData.Y1.iloc[i],
##      testData.VX1.iloc[i],
##      testData.VY1.iloc[i],
##      testData.X2.iloc[i],
##      testData.Y2.iloc[i],
##      testData.VX2.iloc[i],
##      testData.VY2.iloc[i]
##      ])
##
##    yTest.append([
##      testData.X1.iloc[i+1],
##      testData.Y1.iloc[i+1],
##      testData.VX1.iloc[i+1],
##      testData.VY1.iloc[i+1]
##      ])
##
##  elif testData["1to3"].iloc[i] == 1:
##    xTest.append([
##      testData.X1.iloc[i],
##      testData.Y1.iloc[i],
##      testData.VX1.iloc[i],
##      testData.VY1.iloc[i],
##      testData.X3.iloc[i],
##      testData.Y3.iloc[i],
##      testData.VX3.iloc[i],
##      testData.VY3.iloc[i]
##      ])
##
##    yTest.append([
##      testData.X1.iloc[i+1],
##      testData.Y1.iloc[i+1],
##      testData.VX1.iloc[i+1],
##      testData.VY1.iloc[i+1]
##      ])
##
##  elif testData["1to4"].iloc[i] == 1:
##    xTest.append([
##      testData.X1.iloc[i],
##      testData.Y1.iloc[i],
##      testData.VX1.iloc[i],
##      testData.VY1.iloc[i],
##      testData.X4.iloc[i],
##      testData.Y4.iloc[i],
##      testData.VX4.iloc[i],
##      testData.VY4.iloc[i]
##      ])
##
##    yTest.append([
##      testData.X1.iloc[i+1],
##      testData.Y1.iloc[i+1],
##      testData.VX1.iloc[i+1],
##      testData.VY1.iloc[i+1]
##      ])
##
##  elif testData["1to5"].iloc[i] == 1:
##    xTest.append([
##      testData.X1.iloc[i],
##      testData.Y1.iloc[i],
##      testData.VX1.iloc[i],
##      testData.VY1.iloc[i],
##      testData.X5.iloc[i],
##      testData.Y5.iloc[i],
##      testData.VX5.iloc[i],
##      testData.VY5.iloc[i]
##      ])
##
##    yTest.append([
##      testData.X1.iloc[i+1],
##      testData.Y1.iloc[i+1],
##      testData.VX1.iloc[i+1],
##      testData.VY1.iloc[i+1]
##      ])
##
##  elif testData["2to1"].iloc[i] == 1:
##    xTest.append([
##      testData.X2.iloc[i],
##      testData.Y2.iloc[i],
##      testData.VX2.iloc[i],
##      testData.VY2.iloc[i],
##      testData.X1.iloc[i],
##      testData.Y1.iloc[i],
##      testData.VX1.iloc[i],
##      testData.VY1.iloc[i]
##      ])
##
##    yTest.append([
##      testData.X2.iloc[i+1],
##      testData.Y2.iloc[i+1],
##      testData.VX2.iloc[i+1],
##      testData.VY2.iloc[i+1]
##      ])
##
##  elif testData["2to3"].iloc[i] == 1:
##    xTest.append([
##      testData.X2.iloc[i],
##      testData.Y2.iloc[i],
##      testData.VX2.iloc[i],
##      testData.VY2.iloc[i],
##      testData.X3.iloc[i],
##      testData.Y3.iloc[i],
##      testData.VX3.iloc[i],
##      testData.VY3.iloc[i]
##      ])
##
##    yTest.append([
##      testData.X2.iloc[i+1],
##      testData.Y2.iloc[i+1],
##      testData.VX2.iloc[i+1],
##      testData.VY2.iloc[i+1]
##      ])
##  elif testData["2to4"].iloc[i] == 1:
##    xTest.append([
##      testData.X2.iloc[i],
##      testData.Y2.iloc[i],
##      testData.VX2.iloc[i],
##      testData.VY2.iloc[i],
##      testData.X4.iloc[i],
##      testData.Y4.iloc[i],
##      testData.VX4.iloc[i],
##      testData.VY4.iloc[i]
##      ])
##
##    yTest.append([
##      testData.X2.iloc[i+1],
##      testData.Y2.iloc[i+1],
##      testData.VX2.iloc[i+1],
##      testData.VY2.iloc[i+1]
##      ])
##
##  elif testData["2to5"].iloc[i] == 1:
##    xTest.append([
##      testData.X2.iloc[i],
##      testData.Y2.iloc[i],
##      testData.VX2.iloc[i],
##      testData.VY2.iloc[i],
##      testData.X5.iloc[i],
##      testData.Y5.iloc[i],
##      testData.VX5.iloc[i],
##      testData.VY5.iloc[i]
##      ])
##
##    yTest.append([
##      testData.X2.iloc[i+1],
##      testData.Y2.iloc[i+1],
##      testData.VX2.iloc[i+1],
##      testData.VY2.iloc[i+1]
##      ])
##
##  elif testData["2to5"].iloc[i] == 1:
##    xTest.append([
##      testData.X2.iloc[i],
##      testData.Y2.iloc[i],
##      testData.VX2.iloc[i],
##      testData.VY2.iloc[i],
##      testData.X5.iloc[i],
##      testData.Y5.iloc[i],
##      testData.VX5.iloc[i],
##      testData.VY5.iloc[i]
##      ])
##
##    yTest.append([
##      testData.X2.iloc[i+1],
##      testData.Y2.iloc[i+1],
##      testData.VX2.iloc[i+1],
##      testData.VY2.iloc[i+1]
##      ])
##
##  elif testData["2to5"].iloc[i] == 1:
##    xTest.append([
##      testData.X2.iloc[i],
##      testData.Y2.iloc[i],
##      testData.VX2.iloc[i],
##      testData.VY2.iloc[i],
##      testData.X5.iloc[i],
##      testData.Y5.iloc[i],
##      testData.VX5.iloc[i],
##      testData.VY5.iloc[i]
##      ])
##
##    yTest.append([
##      testData.X2.iloc[i+1],
##      testData.Y2.iloc[i+1],
##      testData.VX2.iloc[i+1],
##      testData.VY2.iloc[i+1]
##      ])
##
##  elif testData["3to1"].iloc[i] == 1:
##    xTest.append([
##      testData.X3.iloc[i],
##      testData.Y3.iloc[i],
##      testData.VX3.iloc[i],
##      testData.VY3.iloc[i],
##      testData.X1.iloc[i],
##      testData.Y1.iloc[i],
##      testData.VX1.iloc[i],
##      testData.VY1.iloc[i]
##      ])
##
##    yTest.append([
##      testData.X3.iloc[i+1],
##      testData.Y3.iloc[i+1],
##      testData.VX3.iloc[i+1],
##      testData.VY3.iloc[i+1]
##      ])
##
##  elif testData["3to2"].iloc[i] == 1:
##    xTest.append([
##      testData.X3.iloc[i],
##      testData.Y3.iloc[i],
##      testData.VX3.iloc[i],
##      testData.VY3.iloc[i],
##      testData.X2.iloc[i],
##      testData.Y2.iloc[i],
##      testData.VX2.iloc[i],
##      testData.VY2.iloc[i]
##      ])
##
##    yTest.append([
##      testData.X3.iloc[i+1],
##      testData.Y3.iloc[i+1],
##      testData.VX3.iloc[i+1],
##      testData.VY3.iloc[i+1]
##      ])
##
##  elif testData["3to4"].iloc[i] == 1:
##    xTest.append([
##      testData.X3.iloc[i],
##      testData.Y3.iloc[i],
##      testData.VX3.iloc[i],
##      testData.VY3.iloc[i],
##      testData.X4.iloc[i],
##      testData.Y4.iloc[i],
##      testData.VX4.iloc[i],
##      testData.VY4.iloc[i]
##      ])
##
##    yTest.append([
##      testData.X3.iloc[i+1],
##      testData.Y3.iloc[i+1],
##      testData.VX3.iloc[i+1],
##      testData.VY3.iloc[i+1]
##      ])
##
##xTestFinal = np.array(xTest).reshape(len(xTest),1,8)
##yTestFinal = np.array(yTest).reshape(len(yTest),1,4)
##
##np.save("xTestVision.npy", xTestFinal)
##np.save("yTestVision.npy", yTestFinal)
##


##df = pd.read_csv("training data - five fish.csv", low_memory = False)
##
##trainData = df.iloc[1000:len(df)]
##testData = df.iloc[:1000]
##
##xTrain = []
##yTrain = []
##
##xTest = []
##yTest = []
###1to2	1to3	1to4	1to5	2to1	2to3	2to4
###2to5	3to1	3to2	3to4	3to5	4to1	4to2
###4to3	4to5	5to1	5to2	5to3	5to4
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
##    trainData.VY2.iloc[i],
##    trainData.X3.iloc[i],
##    trainData.Y3.iloc[i],
##    trainData.VX3.iloc[i],
##    trainData.VY3.iloc[i],
##    trainData.X4.iloc[i],
##    trainData.Y4.iloc[i],
##    trainData.VX4.iloc[i],
##    trainData.VY4.iloc[i],
##    trainData.X5.iloc[i],
##    trainData.Y5.iloc[i],
##    trainData.VX5.iloc[i],
##    trainData.VY5.iloc[i]
##    ])
##
##  yTrain.append([
##    trainData["1to2"].iloc[i],
##    trainData["1to3"].iloc[i],
##    trainData["1to4"].iloc[i],
##    trainData["1to5"].iloc[i],
##    trainData["2to1"].iloc[i],
##    trainData["2to3"].iloc[i],
##    trainData["2to4"].iloc[i],
##    trainData["2to5"].iloc[i],
##    trainData["3to1"].iloc[i],
##    trainData["3to2"].iloc[i],
##    trainData["3to4"].iloc[i],
##    trainData["3to5"].iloc[i],
##    trainData["4to1"].iloc[i],
##    trainData["4to2"].iloc[i],
##    trainData["4to3"].iloc[i],
##    trainData["4to5"].iloc[i],
##    trainData["5to1"].iloc[i],
##    trainData["5to2"].iloc[i],
##    trainData["5to3"].iloc[i],
##    trainData["5to4"].iloc[i]
##    ])
##
##xTrainFinal = np.array(xTrain).reshape(len(xTrain),1,20)
##yTrainFinal = np.array(yTrain).reshape(len(yTrain),1,20)
##
##np.save("xTrainCat.npy", xTrainFinal)
##np.save("yTrainCat.npy", yTrainFinal)
##
##for i in range(len(testData.X1) - 2):
##  print(i)
##  xTest.append([
##    testData.X1.iloc[i],
##    testData.Y1.iloc[i],
##    testData.VX1.iloc[i],
##    testData.VY1.iloc[i],
##    testData.X2.iloc[i],
##    testData.Y2.iloc[i],
##    testData.VX2.iloc[i],
##    testData.VY2.iloc[i],
##    testData.X3.iloc[i],
##    testData.Y3.iloc[i],
##    testData.VX3.iloc[i],
##    testData.VY3.iloc[i],
##    testData.X4.iloc[i],
##    testData.Y4.iloc[i],
##    testData.VX4.iloc[i],
##    testData.VY4.iloc[i],
##    testData.X5.iloc[i],
##    testData.Y5.iloc[i],
##    testData.VX5.iloc[i],
##    testData.VY5.iloc[i]
##    ])
##
##  yTest.append([
##    testData["1to2"].iloc[i],
##    testData["1to3"].iloc[i],
##    testData["1to4"].iloc[i],
##    testData["1to5"].iloc[i],
##    testData["2to1"].iloc[i],
##    testData["2to3"].iloc[i],
##    testData["2to4"].iloc[i],
##    testData["2to5"].iloc[i],
##    testData["3to1"].iloc[i],
##    testData["3to2"].iloc[i],
##    testData["3to4"].iloc[i],
##    testData["3to5"].iloc[i],
##    testData["4to1"].iloc[i],
##    testData["4to2"].iloc[i],
##    testData["4to3"].iloc[i],
##    testData["4to5"].iloc[i],
##    testData["5to1"].iloc[i],
##    testData["5to2"].iloc[i],
##    testData["5to3"].iloc[i],
##    testData["5to4"].iloc[i]
##    ])
##
##xTestFinal = np.array(xTest).reshape(len(xTest),1,20)
##yTestFinal = np.array(yTest).reshape(len(yTest),1,20)
##
##np.save("xTestCat.npy", xTestFinal)
##np.save("yTestCat.npy", yTestFinal)
