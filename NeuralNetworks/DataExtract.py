import rdata
import numpy as np
import matplotlib as mpl 
mpl.rcParams["animation.ffmpeg_path"] = r"C:\\PATH_Programs\\ffmpeg.exe"
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib import gridspec
import matplotlib.ticker as plticker
import random

parsed = rdata.parser.parse_file("5FishTest.RData")
converted = rdata.conversion.convert(parsed)
data = converted["data"]


fig = plt.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(1, 1, height_ratios = [1])
ax1 = plt.subplot(gs[0])

x2Frames = data.x2
y2Frames = data.y2

x1Frames = data.x1
y1Frames = data.y1

x3Frames = data.x3
y3Frames = data.y3

x4Frames = data.x4
y4Frames = data.y4

x5Frames = data.x5
y5Frames = data.y5

xBounds = [min(data.x1.min(),data.x2.min(),data.x3.min(),data.x4.min(),data.x5.min())
           ,max(data.x1.max(),data.x2.max(),data.x3.max(),data.x3.max(),data.x5.max())]
yBounds = [min(data.y1.min(),data.y2.min(),data.y3.min(),data.y4.min(),data.y5.min()),
           max(data.y1.max(),data.y2.max(),data.y3.max(),data.y4.max(),data.y5.max())]

def plot1D(k):
  ax1.cla()
  ax1.set_xlim(xBounds)
  ax1.set_ylim(yBounds)
  
  if k < 30:
    ax1.scatter(x1Frames.iloc[:k], y1Frames.iloc[:k], color = "blue", s = 1)

    ax1.scatter(x2Frames.iloc[:k], y2Frames.iloc[:k], color = "red", s = 1)

    ax1.scatter(x3Frames.iloc[:k], y2Frames.iloc[:k], color = "green", s = 1)

    ax1.scatter(x4Frames.iloc[:k], y2Frames.iloc[:k], color = "orange", s = 1)

    ax1.scatter(x5Frames.iloc[:k], y2Frames.iloc[:k], color = "black", s = 1)

    return plt

  else:
    ax1.scatter(x1Frames.iloc[k-30:k], y1Frames.iloc[k-30:k], color = "blue", s = 1)

    ax1.scatter(x2Frames.iloc[k-30:k], y2Frames.iloc[k-30:k], color = "red", s = 1)

    ax1.scatter(x3Frames.iloc[k-30:k], y2Frames.iloc[k-30:k], color = "green", s = 1)

    ax1.scatter(x4Frames.iloc[k-30:k], y2Frames.iloc[k-30:k], color = "orange", s = 1)

    ax1.scatter(x5Frames.iloc[k-30:k], y2Frames.iloc[k-30:k], color = "black", s = 1)
    
    return plt


def animate(k):
  plot1D(k)

frameCount = len(x1Frames)

anim = animation.FuncAnimation(fig, animate, interval=1, frames= frameCount, repeat=False)
writervideo = animation.FFMpegWriter(fps=24) 
anim.save("5FishAnimation.mp4", writer=writervideo)
print("Done Saving")
plt.show()
