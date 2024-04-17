# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:29:48 2024

@author: Hassan Miah
"""

def groupvel(vx1,vy1,vx2,vy2,N):
    zx = (vx1 + vx2)/N
    zy = (vy1 + vy2)/N
    return (zx,zy)
