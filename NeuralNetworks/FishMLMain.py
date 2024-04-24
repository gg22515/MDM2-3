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

numberOfFish = 2

def LoadData(path):
  pass

def GenerateTrainingData(data, numberOfFish):
  pass

def GenerateTestingData(data, numberOfFish):
  pass

class VisionBlock: #Pass in complete dataset, get out vision dataset
  def __init__(self):
    pass

  def Train(self):
    pass

  def Load(self):
    pass

  def Predict(self): #Give everything initially
    pass

class BrainBlock: #Main perceptron
  def __init__(self):
    pass

  def Train(self):
    pass

  def Load(self):
    pass

  def Predict(self):
    pass
