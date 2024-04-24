import rdata
import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib as mpl 
mpl.rcParams["animation.ffmpeg_path"] = r"C:\\PATH_Programs\\ffmpeg.exe"
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib import gridspec
import matplotlib.ticker as plticker

parsed = rdata.parser.parse_file("5FishTrain.RData")
converted = rdata.conversion.convert(parsed)
trainData = converted["data"]

parsed = rdata.parser.parse_file("5FishTest.RData")
converted = rdata.conversion.convert(parsed)
testData = converted["data"]

#Dynamic data frame get keys - SO

print(list(trainData.columns.values))

##model = Sequential()
##model.add(Dense(10, activation = "relu", input_shape = (1,18)))
##model.add(Dense(500, activation = "relu"))
##model.add(Dense(500, activation = "relu"))
##model.add(Dense(500, activation = "relu"))
##model.add(Dense(100, activation = "relu"))
##model.add(Dense(100, activation = "relu"))
##model.add(Dense(2, activation = "linear"))
##
##model.compile(loss = "mse", optimizer = "adam", metrics = ["mse"])
##
##trainingEpochs = 500
##
###Fish 1
##print("Fish 1")
##inputData = []
##outputData = []
##
##for index in range(len(trainData.x1) - 2):
##  inputData.append([
##  trainData.x1.iloc[index],
##  trainData.y1.iloc[index],
##  trainData.x2.iloc[index],
##  trainData.y2.iloc[index],
##  trainData.vx2.iloc[index],
##  trainData.vy2.iloc[index],
##  trainData.x3.iloc[index],
##  trainData.y3.iloc[index],
##  trainData.vx3.iloc[index],
##  trainData.vy3.iloc[index],
##  trainData.x4.iloc[index],
##  trainData.y4.iloc[index],
##  trainData.vx4.iloc[index],
##  trainData.vy4.iloc[index],
##  trainData.x5.iloc[index],
##  trainData.y5.iloc[index],
##  trainData.vx5.iloc[index],
##  trainData.vy5.iloc[index]
##  ])
##
##  outputData.append([
##  trainData.x1.iloc[index + 1],
##  trainData.y1.iloc[index + 1]
##  ])
##
##inputTrainData = np.array(inputData).reshape(len(inputData),1,18)
##outputTrainData = np.array(outputData).reshape(len(outputData),1,2)
##
##history = model.fit(inputTrainData, outputTrainData, epochs = trainingEpochs)
##
###Fish 2
##print("Fish 2")
##inputData = []
##outputData = []
##
##for index in range(len(trainData.x1) - 2):
##  inputData.append([
##  trainData.x2.iloc[index],
##  trainData.y2.iloc[index],
##  trainData.x1.iloc[index],
##  trainData.y1.iloc[index],
##  trainData.vx1.iloc[index],
##  trainData.vy1.iloc[index],
##  trainData.x3.iloc[index],
##  trainData.y3.iloc[index],
##  trainData.vx3.iloc[index],
##  trainData.vy3.iloc[index],
##  trainData.x4.iloc[index],
##  trainData.y4.iloc[index],
##  trainData.vx4.iloc[index],
##  trainData.vy4.iloc[index],
##  trainData.x5.iloc[index],
##  trainData.y5.iloc[index],
##  trainData.vx5.iloc[index],
##  trainData.vy5.iloc[index]
##  ])
##
##  outputData.append([
##  trainData.x2.iloc[index + 1],
##  trainData.y2.iloc[index + 1]
##  ])
##
##inputTrainData = np.array(inputData).reshape(len(inputData),1,18)
##outputTrainData = np.array(outputData).reshape(len(outputData),1,2)
##
##history = model.fit(inputTrainData, outputTrainData, epochs = trainingEpochs)
##
##testInput = []
##testOutput = []
##
###Fish 3
##print("Fish 3")
##inputData = []
##outputData = []
##
##for index in range(len(trainData.x1) - 2):
##  inputData.append([
##  trainData.x3.iloc[index],
##  trainData.y3.iloc[index],
##  trainData.x2.iloc[index],
##  trainData.y2.iloc[index],
##  trainData.vx2.iloc[index],
##  trainData.vy2.iloc[index],
##  trainData.x1.iloc[index],
##  trainData.y1.iloc[index],
##  trainData.vx1.iloc[index],
##  trainData.vy1.iloc[index],
##  trainData.x4.iloc[index],
##  trainData.y4.iloc[index],
##  trainData.vx4.iloc[index],
##  trainData.vy4.iloc[index],
##  trainData.x5.iloc[index],
##  trainData.y5.iloc[index],
##  trainData.vx5.iloc[index],
##  trainData.vy5.iloc[index]
##  ])
##
##  outputData.append([
##  trainData.x3.iloc[index + 1],
##  trainData.y3.iloc[index + 1]
##  ])
##
##inputTrainData = np.array(inputData).reshape(len(inputData),1,18)
##outputTrainData = np.array(outputData).reshape(len(outputData),1,2)
##
##history = model.fit(inputTrainData, outputTrainData, epochs = trainingEpochs)
##
###Fish 4
##print("Fish 4")
##inputData = []
##outputData = []
##
##for index in range(len(trainData.x1) - 2):
##  inputData.append([
##  trainData.x4.iloc[index],
##  trainData.y4.iloc[index],
##  trainData.x2.iloc[index],
##  trainData.y2.iloc[index],
##  trainData.vx2.iloc[index],
##  trainData.vy2.iloc[index],
##  trainData.x3.iloc[index],
##  trainData.y3.iloc[index],
##  trainData.vx3.iloc[index],
##  trainData.vy3.iloc[index],
##  trainData.x1.iloc[index],
##  trainData.y1.iloc[index],
##  trainData.vx1.iloc[index],
##  trainData.vy1.iloc[index],
##  trainData.x5.iloc[index],
##  trainData.y5.iloc[index],
##  trainData.vx5.iloc[index],
##  trainData.vy5.iloc[index]
##  ])
##
##  outputData.append([
##  trainData.x4.iloc[index + 1],
##  trainData.y4.iloc[index + 1]
##  ])
##
##inputTrainData = np.array(inputData).reshape(len(inputData),1,18)
##outputTrainData = np.array(outputData).reshape(len(outputData),1,2)
##
##history = model.fit(inputTrainData, outputTrainData, epochs = trainingEpochs)
##
###Fish 5
##print("Fish 5")
##inputData = []
##outputData = []
##
##for index in range(len(trainData.x1) - 2):
##  inputData.append([
##  trainData.x5.iloc[index],
##  trainData.y5.iloc[index],
##  trainData.x2.iloc[index],
##  trainData.y2.iloc[index],
##  trainData.vx2.iloc[index],
##  trainData.vy2.iloc[index],
##  trainData.x3.iloc[index],
##  trainData.y3.iloc[index],
##  trainData.vx3.iloc[index],
##  trainData.vy3.iloc[index],
##  trainData.x4.iloc[index],
##  trainData.y4.iloc[index],
##  trainData.vx4.iloc[index],
##  trainData.vy4.iloc[index],
##  trainData.x1.iloc[index],
##  trainData.y1.iloc[index],
##  trainData.vx1.iloc[index],
##  trainData.vy1.iloc[index]
##  ])
##
##  outputData.append([
##  trainData.x5.iloc[index + 1],
##  trainData.y5.iloc[index + 1]
##  ])
##
##inputTrainData = np.array(inputData).reshape(len(inputData),1,18)
##outputTrainData = np.array(outputData).reshape(len(outputData),1,2)
##
##history = model.fit(inputTrainData, outputTrainData, epochs = trainingEpochs)
##
##model.save("SimpleFish")
##
###Test
##for index in range(len(testData.x1) - 2):
##  testInput.append([
##  testData.x1.iloc[index],
##  testData.y1.iloc[index],
##  testData.x2.iloc[index],
##  testData.y2.iloc[index],
##  testData.vx2.iloc[index],
##  testData.vy2.iloc[index],
##  testData.x3.iloc[index],
##  testData.y3.iloc[index],
##  testData.vx3.iloc[index],
##  testData.vy3.iloc[index],
##  testData.x4.iloc[index],
##  testData.y4.iloc[index],
##  testData.vx4.iloc[index],
##  testData.vy4.iloc[index],
##  testData.x5.iloc[index],
##  testData.y5.iloc[index],
##  testData.vx5.iloc[index],
##  testData.vy5.iloc[index]
##  ])
##
##  testOutput.append([
##  testData.x1.iloc[index + 1],
##  testData.y1.iloc[index + 1]
##  ])
##
##inputTestData = np.array(testInput).reshape(len(testInput),1,18)
##outputTestData = np.array(testOutput).reshape(len(testOutput),1,2)
##
##prediction = model.predict(inputTestData)
##
##xPredict = []
##yPredict = []
##
##
##for element in prediction:
##  xPredict.append(np.squeeze(element)[0])
##  yPredict.append(np.squeeze(element)[1])
##
##
##fig = plt.figure(figsize=(8, 6)) 
##gs = gridspec.GridSpec(1, 1, height_ratios = [1])
##ax1 = plt.subplot(gs[0])
##
##x1Frames = testData.x1
##y1Frames = testData.y1
##
##x2Frames = testData.x2
##y2Frames = testData.y2
##
##x3Frames = testData.x3
##y3Frames = testData.y3
##
##x4Frames = testData.x4
##y4Frames = testData.y4
##
##x5Frames = testData.x5
##y5Frames = testData.y5
##
##xBounds = [min(testData.x1.min(),testData.x2.min(),testData.x3.min(),testData.x4.min(),testData.x5.min())
##           ,max(testData.x1.max(),testData.x2.max(),testData.x3.max(),testData.x3.max(),testData.x5.max())]
##yBounds = [min(testData.y1.min(),testData.y2.min(),testData.y3.min(),testData.y4.min(),testData.y5.min()),
##           max(testData.y1.max(),testData.y2.max(),testData.y3.max(),testData.y4.max(),testData.y5.max())]
##
##alpha = 0.6
##
##def plot1D(k):
##  ax1.cla()
##  ax1.set_xlim(xBounds)
##  ax1.set_ylim(yBounds)
##  
##  if k < 30:
##    ax1.scatter(x1Frames.iloc[:k], y1Frames.iloc[:k], color = "red", s = 1, label = "Real Fish")
##
##    ax1.scatter(x2Frames.iloc[:k], y2Frames.iloc[:k], color = "blue", s = 1, alpha = alpha)
##
##    ax1.scatter(x3Frames.iloc[:k], y2Frames.iloc[:k], color = "blue", s = 1, alpha = alpha)
##
##    ax1.scatter(x4Frames.iloc[:k], y2Frames.iloc[:k], color = "blue", s = 1, alpha = alpha)
##
##    ax1.scatter(x5Frames.iloc[:k], y2Frames.iloc[:k], color = "blue", s = 1, alpha = alpha)
##
##    ax1.scatter(xPredict[:k], yPredict[:k], color = "green", s = 1, label = "Simulated Fish")
##
##    plt.legend()
##    return plt
##
##  else:
##    ax1.scatter(x1Frames.iloc[k-30:k], y1Frames.iloc[k-30:k], color = "red", s = 1, label = "Real Fish")
##
##    ax1.scatter(x2Frames.iloc[k-30:k], y2Frames.iloc[k-30:k], color = "blue", s = 1, alpha = alpha)
##
##    ax1.scatter(x3Frames.iloc[k-30:k], y2Frames.iloc[k-30:k], color = "blue", s = 1, alpha = alpha)
##
##    ax1.scatter(x4Frames.iloc[k-30:k], y2Frames.iloc[k-30:k], color = "blue", s = 1, alpha = alpha)
##
##    ax1.scatter(x5Frames.iloc[k-30:k], y2Frames.iloc[k-30:k], color = "blue", s = 1, alpha = alpha)
##
##    ax1.scatter(xPredict[k-30:k], yPredict[k-30:k], color = "green", s = 1, label = "Simulated Fish")
##    
##    plt.legend()
##    return plt
##
##
##def animate(k):
##  plot1D(k)
##
##frameCount = len(x1Frames)
##
##anim = animation.FuncAnimation(fig, animate, interval=1, frames= frameCount, repeat=False)
##plt.show()
##
###model.evaluate(inputTestData, outputTestData)
##
    
