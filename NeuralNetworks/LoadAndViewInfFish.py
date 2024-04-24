import rdata
import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
import matplotlib as mpl 
mpl.rcParams["animation.ffmpeg_path"] = r"C:\\PATH_Programs\\ffmpeg.exe"
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib import gridspec
import matplotlib.ticker as plticker

xTestLoad = np.load("xTest5Fish.npy")
yTestLoad = np.load("yTest5Fish.npy")

xTest = xTestLoad[200:500]
yTest = yTestLoad[200:500]

fig = plt.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(1, 1, height_ratios = [1])
ax1 = plt.subplot(gs[0])

x1Frames = xTest[:,:,0]
y1Frames = xTest[:,:,1]

x2Frames = xTest[:,:,4]
y2Frames = xTest[:,:,5]

x3Frames = xTest[:,:,8]
y3Frames = xTest[:,:,9]

x4Frames = xTest[:,:,12]
y4Frames = xTest[:,:,13]

x5Frames = xTest[:,:,16]
y5Frames = xTest[:,:,17]

model = load_model("5FishVelInfluential")

predictedX = []
predictedVX = []
predictedY = []
predictedVY = []

for index in range(len(x1Frames)):
  print(index)
  if index < 1:
    cacheAr = np.copy(xTest)
    cache = np.squeeze(cacheAr[index])[:8]

    predictedX.append(cache[0])
    predictedY.append(cache[1])
    predictedVX.append(cache[2])
    predictedVY.append(cache[3])

    predCache = np.squeeze(model.predict(cache.reshape((1,8))))

    predictedX.append(predCache[0])
    predictedY.append(predCache[1])
    predictedVX.append(predCache[2])
    predictedVY.append(predCache[3])

  elif index >= 1:
    cacheAr = np.copy(xTest)
    cache = np.squeeze(cacheAr[index])[:8]

    cache[0] = predictedX[index]
    cache[1] = predictedY[index]
    cache[2] = predictedVX[index]
    cache[3] = predictedVY[index]

    predCache = np.squeeze(model.predict(cache.reshape((1,8))))

    predictedX.append(predCache[0])
    predictedY.append(predCache[1])
    predictedVX.append(predCache[2])
    predictedVY.append(predCache[3])

model2 = load_model("5FishAllModelVel4")

predictedX2 = []
predictedVX2 = []
predictedY2 = []
predictedVY2 = []

for index in range(len(x1Frames)):
  print(index)
  if index < 1:
    cacheAr = np.copy(xTest)
    cache = np.squeeze(cacheAr[index])

    predictedX2.append(cache[0])
    predictedY2.append(cache[1])
    predictedVX2.append(cache[2])
    predictedVY2.append(cache[3])

    predCache = np.squeeze(model2.predict(cache.reshape((1,20))))

    predictedX2.append(predCache[0])
    predictedY2.append(predCache[1])
    predictedVX2.append(predCache[2])
    predictedVY2.append(predCache[3])

  elif index >= 1:
    cacheAr = np.copy(xTest)
    cache = np.squeeze(cacheAr[index])

    cache[0] = predictedX2[index]
    cache[1] = predictedY2[index]
    cache[2] = predictedVX2[index]
    cache[3] = predictedVY2[index]

    predCache = np.squeeze(model2.predict(cache.reshape((1,20))))

    predictedX2.append(predCache[0])
    predictedY2.append(predCache[1])
    predictedVX2.append(predCache[2])
    predictedVY2.append(predCache[3])




xBounds = [min([
  min(x1Frames),
  min(x2Frames),
  min(x3Frames),
  min(x4Frames),
  min(x5Frames),
  min(predictedX),
  min(predictedX2)
  ]),
           max([
             max(x1Frames),
             max(x2Frames),
             max(x3Frames),
             max(x4Frames),
             max(x5Frames),
             max(predictedX),
             max(predictedX2)
             ])
           ]

yBounds = [min([
  min(y1Frames),
  min(y2Frames),
  min(y3Frames),
  min(y4Frames),
  min(y5Frames),
  min(predictedY),
  min(predictedY2)
  ]),
           max([
             max(y1Frames),
             max(y2Frames),
             max(y3Frames),
             max(y4Frames),
             max(y5Frames),
             max(predictedY),
             max(predictedY2)
             ])
           ]


def plot1D(k):
  ax1.cla()
  ax1.set_xlim(xBounds)
  ax1.set_ylim(yBounds)
  
  if k < 30:
    ax1.scatter(x1Frames[:k], y1Frames[:k], color = "green", s = 1, label = "Real Fish")
    ax1.scatter(x2Frames[:k], y2Frames[:k], color = "purple", s = 1, label = "Real Fish")
    ax1.scatter(x3Frames[:k], y3Frames[:k], color = "blue", s = 1, label = "Real Fish")
    ax1.scatter(x4Frames[:k], y4Frames[:k], color = "blue", s = 1, label = "Real Fish")
    ax1.scatter(x5Frames[:k], y5Frames[:k], color = "blue", s = 1, label = "Real Fish")

    ax1.scatter(predictedX[:k], predictedY[:k], color = "red", s = 1, label = "Inf. Neigh. Fish")

    ax1.scatter(predictedX2[:k], predictedY2[:k], color = "black", s = 1, label = "All Seeing Fish")

    plt.legend()
    return plt

  else:
    ax1.scatter(x1Frames[k-30:k], y1Frames[k-30:k], color = "green", s = 1, label = "Real (starting) Fish")
    ax1.scatter(x2Frames[k-30:k], y2Frames[k-30:k], color = "purple", s = 1, label = "Real  (followed) Fish")
    ax1.scatter(x3Frames[k-30:k], y3Frames[k-30:k], color = "blue", s = 1, label = "Real Fish")
    ax1.scatter(x4Frames[k-30:k], y4Frames[k-30:k], color = "blue", s = 1, label = "Real Fish")
    ax1.scatter(x5Frames[k-30:k], y5Frames[k-30:k], color = "blue", s = 1, label = "Real Fish")

    ax1.scatter(predictedX[k-30:k], predictedY[k-30:k], color = "red", s = 1, label = "Inf. Neigh. Fish")

    ax1.scatter(predictedX2[k-30:k], predictedY2[k-30:k], color = "black", s = 1, label = "All Seeing Fish")
    
    plt.legend()
    return plt


def animate(k):
  plot1D(k)

frameCount = len(x1Frames)

#anim = animation.FuncAnimation(fig, animate, interval=1, frames= frameCount, repeat=False)
#plt.show()


##df = pd.read_csv("5FishMainDataCleaned.csv", low_memory = False)
##testData = df.iloc[:1000,:]
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
##  for i in range(len(testData.X1) - 2):
##    xTest.append([
##      testData["X" + currentFish].iloc[i],
##      testData["Y" + currentFish].iloc[i],
##      testData["VX" + currentFish].iloc[i],
##      testData["VY" + currentFish].iloc[i],
##      testData["X" + fishCache[0]].iloc[i],
##      testData["Y" + fishCache[0]].iloc[i],
##      testData["VX" + fishCache[0]].iloc[i],
##      testData["VY" + fishCache[0]].iloc[i],
##      testData["X" + fishCache[1]].iloc[i],
##      testData["Y" + fishCache[1]].iloc[i],
##      testData["VX" + fishCache[1]].iloc[i],
##      testData["VY" + fishCache[1]].iloc[i],
##      testData["X" + fishCache[2]].iloc[i],
##      testData["Y" + fishCache[2]].iloc[i],
##      testData["VX" + fishCache[2]].iloc[i],
##      testData["VY" + fishCache[2]].iloc[i],
##      testData["X" + fishCache[3]].iloc[i],
##      testData["Y" + fishCache[3]].iloc[i],
##      testData["VX" + fishCache[3]].iloc[i],
##      testData["VY" + fishCache[3]].iloc[i]
##      ])
##    
##    yTest.append([
##      testData["X" + currentFish].iloc[i+1],
##      testData["Y" + currentFish].iloc[i+1],
##      testData["VX" + currentFish].iloc[i+1],
##      testData["VY" + currentFish].iloc[i+1]
##      ])
##
##xTestFinal = np.array(xTest).reshape(len(xTest),1,20)
##yTestFinal = np.array(yTest).reshape(len(yTest),1,4)
##
##np.save("xTest5Fish.npy", xTestFinal)
##np.save("yTest5Fish.npy", yTestFinal)
