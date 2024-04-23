# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 23:56:38 2024

@author: Hassan Miah
"""
import numpy as np

#Calculating the group velocity centroid for each frame of dataset called
def groupvel(data):
    group_x = np.array([]) #Initialise x component of velocity array
    group_y = np.array([]) #Initialise y component of velocity array
    
    #Calculating the x component
    for num in range(0,len(data[:,1])):
        x_val = (data[num,15]+data[num,19] + data[num,23] + data[num,27] + \
            data[num,31])/5
        group_x = np.append(group_x,x_val)
    #Calculating the y component
        y_val = (data[num,16] + data[num,20] + data[num,24] + data[num,28] + \
            data[num,32])/5
        group_y = np.append(group_y,y_val)
    return group_x,group_y

        
        
            