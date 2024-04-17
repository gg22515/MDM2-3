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
import moveav as mov
import FunctionInfluential as inf
import groupvelocity as grp

turning_rank_1 = 0
turning_rank_2 = 0
same = 0
    
files_to_read = ["2\exp02H20141128_16h06.csv","2\exp02H20141127_14h13.csv", \
                 "2\exp02H20141127_16h29.csv", "2\exp02H20141204_16h01.csv", \
                     "2\exp02H20141204_16h01.csv" ,"2\exp02H20141204_17h28.csv", \
                         "2\exp02H20141205_16h11.csv","2\exp02H20141205_17h37.csv", \
                             "2\exp02H20141211_11h11.csv","2\exp02H20141212_15h45.csv", \
                                 "2\exp02H20141212_17h12.csv"]


for file in files_to_read:
    df = pd.read_csv(file)

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
    vx1 = df.iloc[:, 6]
    vy1 = df.iloc[:, 7]
    vx2 = df.iloc[:, 8]
    vy2 = df.iloc[:, 9]
    headings1 = df.iloc[:, 2]
    headings2 = df.iloc[:, 5]
    uvx1 = df.iloc[:, 10]
    uvy1 = df.iloc[:, 11]
    uvx2 = df.iloc[:, 12]
    uvy2 = df.iloc[:, 13]

    # Convert remaining values to floats
    vx1 = vx1.astype(float)
    vy1 = vy1.astype(float)
    vx2 = vx2.astype(float)
    vy2 = vy2.astype(float)

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
    for val in range(0,len(x1)):
        angle1 = inc.incidence(x1[val],y1[val],headings1[val])
        angle2 = inc.incidence(x2[val],y2[val],headings2[val])
        incidences1 = np.append(incidences1,angle1)
        incidences2 = np.append(incidences2,angle2)

    incidences1 = np.mod(incidences1 + np.pi, 2 * np.pi) - np.pi
    incidences2 = np.mod(incidences2 + np.pi, 2 * np.pi) - np.pi

    turningpoints = np.array([]) #EXACT FRAME THE U-TURN TAKES PLACE
    for tu in range(0,len(x1)-1):
        if incidences1[tu] * incidences1[tu+1] < 0:
            turningpoints = np.append(turningpoints,tu)
    to_remove = []
    for l in range(len(turningpoints)):
        if turningpoints[l] < 100:
            to_remove.append(l)
        if abs(turningpoints[l] - len(x1)) < 100:
            to_remove.append(l)
    
    turningpoints = np.delete(turningpoints,to_remove)
    
            
    for pos in range(len(turningpoints)):
        x1st = x1[int(turningpoints[pos])-100:int(turningpoints[pos])]
        y1st = y1[int(turningpoints[pos])-100:int(turningpoints[pos])]
        vx1st = vx1[int(turningpoints[pos])-100:int(turningpoints[pos])]
        x2st = x2[int(turningpoints[pos])-100:int(turningpoints[pos])]
        y2st = y2[int(turningpoints[pos])-100:int(turningpoints[pos])]
        vx2st = vx2[int(turningpoints[pos])-100:int(turningpoints[pos])]
        vy2st = vy2[int(turningpoints[pos])-100:int(turningpoints[pos])]
        incst1 = incidences1[int(turningpoints[pos])-100:int(turningpoints[pos])]
        incst2 = incidences2[int(turningpoints[pos])-100:int(turningpoints[pos])]
        
        x1en = x1[int(turningpoints[pos]):int(turningpoints[pos])+100]
        y1en = y1[int(turningpoints[pos]):int(turningpoints[pos])+100]
        x2en = x2[int(turningpoints[pos]):int(turningpoints[pos])+100]
        y2en = y2[int(turningpoints[pos]):int(turningpoints[pos])+100]
        incen1 = incidences1[int(turningpoints[pos]):int(turningpoints[pos])+100]
        incen2 = incidences2[int(turningpoints[pos]):int(turningpoints[pos])+100]
        
        uturnstart1 = mov.moveav(incst1,5)
        uturnstart2 = mov.moveav(incst2,5)
        
        corr_values = inf.influential(file,int(turningpoints[pos])-100,100)
        
        if uturnstart1 == uturnstart2:
            same +=1
        if uturnstart1 < uturnstart2:
            if corr_values[uturnstart1] < 0.95:
                turning_rank_1 +=1
            else:
                turning_rank_2 +=1
        if uturnstart1 > uturnstart2:
            if corr_values[uturnstart1] > 0.95:
                turning_rank_1 +=1
            else:
                turning_rank_2 +=1
                
        z = grp.groupvel(vx1,vy1,vx2,vy2,2)
        
        
        
            
        
        
        
        






