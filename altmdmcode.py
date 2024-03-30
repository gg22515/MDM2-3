# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:27:04 2024

@author: Hassan Miah
"""

import pandas as pd
import numpy as np
import polarisation2 as p2
import matplotlib.pyplot as plt
import incidence as inc
import turnangle as tna


df = pd.read_csv("2\exp02H20141128_16h06.csv")

invalid_strings = ['#DIV/0!', 'NA', '#VALUE!']
for col in df.columns:
    for invalid_str in invalid_strings:
        df = df[df[col] != invalid_str]

# Convert remaining values to floats
df = df.apply(pd.to_numeric, errors='ignore')

# Drop rows with NaN values
df = df.dropna()

# Select relevant columns
x1 = df.iloc[:, 0]
y1 = df.iloc[:, 1]
x2 = df.iloc[:, 3]
y2 = df.iloc[:, 4]
headings1 = df.iloc[:, 2]
headings2 = df.iloc[:, 5]
uvx1 = df.iloc[:, 10]
uvy1 = df.iloc[:, 11]
uvx2 = df.iloc[:, 12]
uvy2 = df.iloc[:, 13]
n = 10000
# Trim the data to first 500 rows
x1 = x1[:n]
y1 = y1[:n]
x2 = x2[:]
y2 = y2[:n]
headings1 = headings1[:n]
headings2 = headings2[:n]
uvx1 = uvx1[:n]
uvy1 = uvy1[:n]
uvx2 = uvx2[:n]
uvy2 = uvy2[:n]

# Convert remaining values to floats
uvx1 = uvx1.astype(float)
uvy1 = uvy1.astype(float)
uvx2 = uvx2.astype(float)
uvy2 = uvy2.astype(float)

x1 = np.array(x1)
x2 = np.array(x2)
y1 = np.array(y1)
y2 = np.array(y2)
headings1 = np.array(headings1)
headings2 = np.array(headings2)
uvx1 = np.array(uvx1)
uvy1 = np.array(uvy1)
uvx2 = np.array(uvx2)
uvy2 = np.array(uvy2)


incidences1 = np.array([])
incidences2 = np.array([])
for val in range(0,n):
    angle1 = inc.incidence(x1[val],y1[val],headings1[val])
    angle2 = inc.incidence(x2[val],y2[val],headings2[val])
    incidences1 = np.append(incidences1,angle1)
    incidences2 = np.append(incidences2,angle2)

incidences1 = np.mod(incidences1 + np.pi, 2 * np.pi) - np.pi
incidences2 = np.mod(incidences2 + np.pi, 2 * np.pi) - np.pi

turningpoints = np.array([]) #EXACT FRAME THE U-TURN TAKES PLACE
for tu in range(0,n-1):
    if incidences1[tu] * incidences1[tu+1] < 0:
        turningpoints = np.append(turningpoints,tu)


t = np.linspace(1,n,n)

#Turning Angles

turning_angles = np.array([])
for val in range(2,n):
    turnangle = tna.turnangle(x1[val-2],x1[val-1],x1[val],y1[val-2],y1[val-1],y1[val])
    turning_angles = np.append(turning_angles,turnangle)


for i in range(0,len(turning_angles)-1):
    if turning_angles[i] < np.pi/3:
        first_dist = turning_angles[:i]
        
        
# plt.hist(turning_angles,bins = 100, color = 'red')
# plt.xlabel('Turning Angles')
# plt.ylabel('Frequency')
# plt.title('Turning Angles of Fish 1')
# plt.show()




