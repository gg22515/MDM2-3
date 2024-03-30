# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:31:53 2024

@author: Hassan Miah
"""

import math as m
import numpy as np
def incidence(x1,y1,h1):
    angle = np.arctan(y1/x1)
    incid = h1 - angle
    return incid


    