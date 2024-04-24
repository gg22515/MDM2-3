import rdata
import math
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

parsed = rdata.parser.parse_file("5FishTest.RData")
converted = rdata.conversion.convert(parsed)
data = converted["data"]


theta = 3.1415


def GetClosestVisFish(data):
  inputData = []
  outputData = []
  for i in range(data.shape[0] - 1):
    cacheDict = {}
    cacheDict[2] = (abs(data.x1.iloc[i] - data.x2.iloc[i])**2 + abs(data.y1.iloc[i] - data.y2.iloc[i])**2)**(1/2)
    cacheDict[3] = (abs(data.x1.iloc[i] - data.x3.iloc[i])**2 + abs(data.y1.iloc[i] - data.y3.iloc[i])**2)**(1/2)
    cacheDict[4] = (abs(data.x1.iloc[i] - data.x4.iloc[i])**2 + abs(data.y1.iloc[i] - data.y4.iloc[i])**2)**(1/2)
    cacheDict[5] = (abs(data.x1.iloc[i] - data.x5.iloc[i])**2 + abs(data.y1.iloc[i] - data.y5.iloc[i])**2)**(1/2)
    
    distanceVector = np.array([data.x2.iloc[i] - data.x1.iloc[i], data.y2.iloc[i] - data.y1.iloc[i]])
    headingVector = np.array([np.cos(data.head1.iloc[i]), np.sin(data.head1.iloc[i])])
    angle = np.arccos(np.clip(np.dot(distanceVector / np.linalg.norm(distanceVector),
                                     headingVector / np.linalg.norm(headingVector)), -1.0, 1.0))
    if abs(angle) < theta:
      cacheDict[2] = (abs(data.x1.iloc[i] - data.x2.iloc[i])**2 + abs(data.y1.iloc[i] - data.y2.iloc[i])**2)**(1/2)

    distanceVector = np.array([data.x3.iloc[i] - data.x1.iloc[i], data.y3.iloc[i] - data.y1.iloc[i]])
    headingVector = np.array([np.cos(data.head1.iloc[i]), np.sin(data.head1.iloc[i])])
    angle = np.arccos(np.clip(np.dot(distanceVector / np.linalg.norm(distanceVector),
                                     headingVector / np.linalg.norm(headingVector)), -1.0, 1.0))
    if abs(angle) < theta:
      cacheDict[3] = (abs(data.x1.iloc[i] - data.x3.iloc[i])**2 + abs(data.y1.iloc[i] - data.y3.iloc[i])**2)**(1/2)

    distanceVector = np.array([data.x4.iloc[i] - data.x1.iloc[i], data.y4.iloc[i] - data.y1.iloc[i]])
    headingVector = np.array([np.cos(data.head1.iloc[i]), np.sin(data.head1.iloc[i])])
    angle = np.arccos(np.clip(np.dot(distanceVector / np.linalg.norm(distanceVector),
                                     headingVector / np.linalg.norm(headingVector)), -1.0, 1.0))
    if abs(angle) < theta:
      cacheDict[4] = (abs(data.x1.iloc[i] - data.x4.iloc[i])**2 + abs(data.y1.iloc[i] - data.y4.iloc[i])**2)**(1/2)

    distanceVector = np.array([data.x5.iloc[i] - data.x1.iloc[i], data.y5.iloc[i] - data.y1.iloc[i]])
    headingVector = np.array([np.cos(data.head1.iloc[i]), np.sin(data.head1.iloc[i])])
    angle = np.arccos(np.clip(np.dot(distanceVector / np.linalg.norm(distanceVector),
                                     headingVector / np.linalg.norm(headingVector)), -1.0, 1.0))
    if abs(angle) < theta:
      cacheDict[5] = (abs(data.x1.iloc[i] - data.x5.iloc[i])**2 + abs(data.y1.iloc[i] - data.y5.iloc[i])**2)**(1/2)
      
    try:  
      closestFish = min(cacheDict, key=cacheDict.get) - 1

    except:
      closestFish = min(cacheDict, key=cacheDict.get) - 1
      print("PING")

    cacheOut = np.array(data.iloc[i][2:16 ])

    cacheOut = np.append(cacheOut,np.array(data.iloc[i][2 + (closestFish*14): 16 + (closestFish*14)]))

    inputData.append(cacheOut)

    outputData.append(np.array(data.iloc[i+1][2:16 ]))

  return inputData, outputData


test = GetClosestVisFish(data)

##headingFrames1 = data.head2
##headingFrames2 = data.head1
##
##def NaiveVision(data):
##  trainInput = []
##  trainOutput = []
##  return trainInput, trainOutput
##
##
##x1Frames = data.x1
##y1Frames = data.y1
##
##x2Frames = data.x2
##y2Frames = data.y2
##
##fig = plt.figure(figsize=(8, 6)) 
##gs = gridspec.GridSpec(1, 1, height_ratios = [1])
##ax1 = plt.subplot(gs[0])
##
##xBounds = [min(data.x1.min(),data.x2.min())
##           ,max(data.x1.max(),data.x2.max())]
##yBounds = [min(data.y1.min(),data.y2.min()),
##           max(data.y1.max(),data.y2.max())]
##
##yBounds = [-1000,1000]
##xBounds = [-1000,1000]
##
##def plot1D(k):
##  #print(x2Frames.iloc[:k], y2Frames.iloc[:k])
##  ax1.cla()
##  ax1.set_xlim(xBounds)
##  ax1.set_ylim(yBounds)
##
##
##  v = 2000
##
##  theta = 0.8
##  
##  if k < 30:
##    ax1.scatter(x1Frames.iloc[:k], y1Frames.iloc[:k], color = "blue", s = 1)
##
##    ax1.scatter(x2Frames.iloc[:k], y2Frames.iloc[:k], color = "red", s = 1)
##
##    #ax1.plot([x1Frames.iloc[k], x1Frames.iloc[k] + v * np.cos(headingFrames.iloc[k])], [y1Frames.iloc[k], y1Frames.iloc[k] + v * np.sin(headingFrames.iloc[k])])
##
##    ax1.plot([x1Frames.iloc[k], x1Frames.iloc[k] + v * np.cos(headingFrames1.iloc[k] + theta)],
##             [y1Frames.iloc[k], y1Frames.iloc[k] + v * np.sin(headingFrames1.iloc[k] + theta)], color = "blue")
##
##    ax1.plot([x1Frames.iloc[k], x1Frames.iloc[k] + v * np.cos(headingFrames1.iloc[k] - theta)],
##             [y1Frames.iloc[k], y1Frames.iloc[k] + v * np.sin(headingFrames1.iloc[k] - theta)], color = "blue")
##    
##
##  elif k >= 30:
##    ax1.scatter(x1Frames.iloc[k-30:k], y1Frames.iloc[k-30:k], color = "blue", s = 1)
##
##    ax1.scatter(x2Frames.iloc[k-30:k], y2Frames.iloc[k-30:k], color = "red", s = 1)
##
##    #ax1.plot([x1Frames.iloc[k], x1Frames.iloc[k] + v * np.cos(headingFrames.iloc[k])], [y1Frames.iloc[k], y1Frames.iloc[k] + v * np.sin(headingFrames.iloc[k])])
##
##    ax1.plot([x1Frames.iloc[k], x1Frames.iloc[k] + v * np.cos(headingFrames1.iloc[k] + theta)],
##             [y1Frames.iloc[k], y1Frames.iloc[k] + v * np.sin(headingFrames1.iloc[k] + theta)], color = "blue")
##
##    ax1.plot([x1Frames.iloc[k], x1Frames.iloc[k] + v * np.cos(headingFrames1.iloc[k] - theta)],
##             [y1Frames.iloc[k], y1Frames.iloc[k] + v * np.sin(headingFrames1.iloc[k] - theta)], color = "blue")
##
##  titleStr = "Heading: " + str(round(np.degrees(headingFrames1.iloc[k]),1))
##
##  #distance  = (abs(x1Frames.iloc[k] - x2Frames.iloc[k])**2 + abs(y1Frames.iloc[k] - y2Frames.iloc[k])**2)**(1/2)
##  distanceVector = np.array([x2Frames.iloc[k] - x1Frames.iloc[k], y2Frames.iloc[k] - y1Frames.iloc[k]])
##  headingVector = np.array([np.cos(headingFrames1.iloc[k]), np.sin(headingFrames1.iloc[k])])
##
##  angle = np.arccos(np.clip(np.dot(distanceVector / np.linalg.norm(distanceVector),
##                                   headingVector / np.linalg.norm(headingVector)), -1.0, 1.0))
##
##  #print(abs(angle), theta)
##
##  if abs(angle) < theta:
##    titleStr = titleStr + " " + "SEEN FISH AT: " + " " +  str(round(x1Frames.iloc[k],1)) + " " + str(round(y1Frames.iloc[k],1))
##
##  else:
##    titleStr = titleStr + " " +  "NO FISH SEEN"
##
##
##  ax1.set_title(titleStr)
##  return plt
##
##
##def animate(k):
##  plot1D(k)
##
##frameCount = len(x1Frames)
##
##anim = animation.FuncAnimation(fig, animate, interval=1, frames= frameCount, repeat=False)
###writervideo = animation.FFMpegWriter(fps=24) 
###anim.save("FishUTurn50.mp4", writer=writervideo)
###print("Done Saving")
##plt.show()
