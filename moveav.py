# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 23:28:39 2024

@author: Hassan Miah
"""
import numpy as np
import math as m

def moveav(incidence, window_size, i=0):
    
    moving_averages = np.array([])
    while i < len(incidence) - window_size + 1:
        window = incidence[i : i + window_size]
        window_average = round(sum(window) / window_size, 2)
        moving_averages = np.append(moving_averages, window_average)
        
        i +=1
    threshold = max(abs(moving_averages))
    for value in range(0,len(incidence)):
        if abs(incidence[value]) > threshold:
            start = value
            break
    return value



