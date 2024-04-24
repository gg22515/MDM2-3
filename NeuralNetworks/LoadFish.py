import rdata
import numpy as np
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

parsed = rdata.parser.parse_file("5FishNaive.RData")
converted = rdata.conversion.convert(parsed)
testData = converted["data"]

model = load_model("SimpleFish")

testInput = []
testOutput = []

for index in range(len(testData.x1) - 2):
  testInput.append([
  testData.x1.iloc[index],
  testData.y1.iloc[index],
  testData.x2.iloc[index],
  testData.y2.iloc[index],
  testData.vx2.iloc[index],
  testData.vy2.iloc[index],
  testData.x3.iloc[index],
  testData.y3.iloc[index],
  testData.vx3.iloc[index],
  testData.vy3.iloc[index],
  testData.x4.iloc[index],
  testData.y4.iloc[index],
  testData.vx4.iloc[index],
  testData.vy4.iloc[index],
  testData.x5.iloc[index],
  testData.y5.iloc[index],
  testData.vx5.iloc[index],
  testData.vy5.iloc[index]
  ])

  testOutput.append([
  testData.x1.iloc[index + 1],
  testData.y1.iloc[index + 1]
  ])

inputTestData = np.array(testInput).reshape(len(testInput),1,18)
outputTestData = np.array(testOutput).reshape(len(testOutput),1,2)

prediction = model.predict(inputTestData)

xPredict = []
yPredict = []


for element in prediction:
  xPredict.append(np.squeeze(element)[0])
  yPredict.append(np.squeeze(element)[1])


fig = plt.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(1, 1, height_ratios = [1])
ax1 = plt.subplot(gs[0])

x1Frames = testData.x1
y1Frames = testData.y1

x2Frames = testData.x2
y2Frames = testData.y2

x3Frames = testData.x3
y3Frames = testData.y3

x4Frames = testData.x4
y4Frames = testData.y4

x5Frames = testData.x5
y5Frames = testData.y5

xBounds = [min(testData.x1.min(),testData.x2.min(),testData.x3.min(),testData.x4.min(),testData.x5.min(), min(xPredict))
           ,max(testData.x1.max(),testData.x2.max(),testData.x3.max(),testData.x3.max(),testData.x5.max(), max(xPredict))]
yBounds = [min(testData.y1.min(),testData.y2.min(),testData.y3.min(),testData.y4.min(),testData.y5.min(), min(yPredict)),
           max(testData.y1.max(),testData.y2.max(),testData.y3.max(),testData.y4.max(),testData.y5.max(), max(yPredict))]

alpha = 0.6

def plot1D(k):
  ax1.cla()
  ax1.set_xlim(xBounds)
  ax1.set_ylim(yBounds)
  
  if k < 30:
    ax1.scatter(x1Frames.iloc[:k], y1Frames.iloc[:k], color = "blue", s = 1, label = "Real Fish")

    ax1.scatter(x2Frames.iloc[:k], y2Frames.iloc[:k], color = "blue", s = 1, alpha = alpha)

    ax1.scatter(x3Frames.iloc[:k], y3Frames.iloc[:k], color = "blue", s = 1, alpha = alpha)

    ax1.scatter(x4Frames.iloc[:k], y4Frames.iloc[:k], color = "blue", s = 1, alpha = alpha)

    ax1.scatter(x5Frames.iloc[:k], y5Frames.iloc[:k], color = "blue", s = 1, alpha = alpha)

    ax1.scatter(xPredict[:k], yPredict[:k], color = "red", s = 1, label = "Simulated Fish")

    plt.legend()
    return plt

  else:
    ax1.scatter(x1Frames.iloc[k-30:k], y1Frames.iloc[k-30:k], color = "blue", s = 1, label = "Real Fish")

    ax1.scatter(x2Frames.iloc[k-30:k], y2Frames.iloc[k-30:k], color = "blue", s = 1, alpha = alpha)

    ax1.scatter(x3Frames.iloc[k-30:k], y3Frames.iloc[k-30:k], color = "blue", s = 1, alpha = alpha)

    ax1.scatter(x4Frames.iloc[k-30:k], y4Frames.iloc[k-30:k], color = "blue", s = 1, alpha = alpha)

    ax1.scatter(x5Frames.iloc[k-30:k], y5Frames.iloc[k-30:k], color = "blue", s = 1, alpha = alpha)

    ax1.scatter(xPredict[k-30:k], yPredict[k-30:k], color = "red", s = 1, label = "Simulated Fish")
    
    plt.legend()
    return plt


def animate(k):
  plot1D(k)

frameCount = len(x1Frames)

anim = animation.FuncAnimation(fig, animate, interval=1, frames= frameCount, repeat=False)
plt.show()


    
