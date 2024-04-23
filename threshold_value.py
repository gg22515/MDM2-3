# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 23:28:39 2024

@author: Hassan Miah
"""
import numpy as np
import math as m

#Calculating the threshold theta value for a U-turn starting or ending
def moveav(incidence, window_size=5, i=0):
    
    moving_averages = np.array([])
    
    #Looping through each of the windows
    while i < len(incidence) - window_size + 1:
        window = incidence[i : i + window_size]
        window_average = round(sum(window) / window_size, 2)
        moving_averages = np.append(moving_averages, window_average)
        
        i +=1
    #Determing threshold for angle of incidence for which a U-turn is 
    #deemed to have begun/ended
    threshold = max(abs(moving_averages))
    
    #checks if the incidence value in the corresponding frame exceeds the 
    #threshold value
    for value in range(0,len(incidence)):
        if abs(incidence[value]) > threshold:
            start = value
            break
    return value #returns the start/end point of U-turn



