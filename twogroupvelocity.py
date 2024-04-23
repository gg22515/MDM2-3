# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:29:48 2024

@author: Hassan Miah
"""

import numpy as np

def groupvel(vx1,vy1,vx2,vy2):
    group_x = np.array([])
    group_y = np.array([])
    for num in range(0,len(vx1)):
        x_val = (vx1[num] + vx2[num])/2
        group_x = np.append(group_x,x_val)
        
        y_val = (vy1[num] + vy2[num])/2
        group_y = np.append(group_y,y_val)
    
    return group_x,group_y

        
