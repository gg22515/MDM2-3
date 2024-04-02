#%%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def normalize_heading(heading):
    """Normalizes a heading to be between 0 and 2*pi."""
    return heading % (2 * np.pi)

def incidence(x1, y1, h1, x2, y2, h2):
    """Calculates the angle of incidence from the perspective of fish 1."""
    dx = x2 - x1
    dy = y2 - y1 
    angle_between_fish = np.arctan2(dy, dx)
    angle_of_incidence = h1 - angle_between_fish 
    return np.arctan2(np.sin(angle_of_incidence), np.cos(angle_of_incidence)) 

leader_tracking = []

def detect_uturns(data_file, start_row, incidence_threshold, heading_threshold, reaction_window=15):
    """Detects U-turns and identifies potential leaders."""
    df = pd.read_csv(data_file, skiprows=start_row, names=['X1','Y1','H1','X2','Y2','H2','VX1','VY1','VX2','VY2','UVX1','UVY1','UVX2','UVY2'])

    df['H1_norm'] = df['H1'].apply(normalize_heading)
    df['H2_norm'] = df['H2'].apply(normalize_heading)

    for i in range(1, len(df)):
        potential_leader = 1
        current_incidence = incidence(df['X1'][i], df['Y1'][i], df['H1_norm'][i], 
                                      df['X2'][i], df['Y2'][i], df['H2_norm'][i])
        prev_incidence = incidence(df['X1'][i-1], df['Y1'][i-1], df['H1_norm'][i-1], 
                                   df['X2'][i-1], df['Y2'][i-1], df['H2_norm'][i-1])

        incidence_change = abs(current_incidence - prev_incidence)
        heading_change = abs(df['H1_norm'][i] - df['H1_norm'][i-1])

        if incidence_change > incidence_threshold and heading_change > heading_threshold:
            u_turn_index = i
            
            

            for j in range(u_turn_index + 1, min(u_turn_index + reaction_window, len(df))):
                follower_incidence = incidence(df['X2'][j], df['Y2'][j], df['H2_norm'][j], 
                                               df['X1'][j], df['Y1'][j], df['H1_norm'][j])
                if abs(follower_incidence) > incidence_threshold:
                    time_delay = j - u_turn_index
                    if time_delay > 10: 
                        potential_leader = 2
                        break
            leader_tracking.append(potential_leader) 
            print("Potential U-turn at index:", u_turn_index, "Potential Leader: Fish", potential_leader)
    leader_counts = Counter(leader_tracking)
    print("Fish 1 leadership count:", leader_counts[1])
    print("Fish 2 leadership count:", leader_counts[2])

# *** Data Loading ***
data_file = "C:/Users/TheBuild/OneDrive - University of Bristol/Documents/University/Year 2/MDM2/Project3/FishData/exp02H20141128_16h06updated.csv"
start_row = 1 
incidence_threshold = 0.8 
heading_threshold = np.pi/2 

detect_uturns(data_file, start_row, incidence_threshold, heading_threshold) 