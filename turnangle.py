# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 18:04:09 2024

@author: Hassan Miah
"""

import numpy as np

def turnangle(x1,x2,x3,y1,y2,y3):
    if (y2-y1) < 0:
        signa = -1
    else:
        signa = 1
    if (y3-y2) < 0:
        signb = -1
    else:
        signb = 1
    alpha = signa*np.arccos((x2-x1)/np.sqrt((x2-x1)**2 + (y2-y1)**2))
    beta = signb*np.arccos((x3-x2)/np.sqrt((x3-x2)**2 + (y3-y2)**2))
    turn_angle = alpha - beta
    return turn_angle