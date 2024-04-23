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
#Intialising lists for turning and position ranks for later
turning_rank_1 = 0
turning_rank_2 = 0
pos_rank = np.array([])

#Calling file with correlation values for all pairwise combinations of fish
other_file = "Corr_values2fish.csv"
myDF = pd.read_csv(other_file)
corr_values = myDF.values

#Obtaining data for fish
file = "2\exp02H20141128_16h06.csv"
df = pd.read_csv(file)

#Cleaning the data of any invalid strings
invalid_strings = ['#DIV/0!', 'NA', '#VALUE!']
for col in df.columns:
    for invalid_str in invalid_strings:
        df = df[df[col] != invalid_str]

# Convert remaining values to floats
df = df.apply(pd.to_numeric, errors='ignore')

# Drop rows with NaN values
df = df.dropna()
df = df.astype(float)
data = df.values

#Initialising angle of incidence arrays for each fish
incidences1 = np.array([])
incidences2 = np.array([])

#Calculating angles for incidences over entire dataset
for val in range(0,len(data[:,0])):
    angle1 = inc.incidence(data[val,0],data[val,1],data[val,2])
    angle2 = inc.incidence(data[val,3],data[val,4],data[val,5])
    incidences1 = np.append(incidences1,angle1)
    incidences2 = np.append(incidences2,angle2)

#All incidences must be in range [-pi,pi]
incidences1 = np.mod(incidences1 + np.pi, 2 * np.pi) - np.pi
incidences2 = np.mod(incidences2 + np.pi, 2 * np.pi) - np.pi

#Calculating turning points by looking for changes in sign
turningpoints1 = np.array([]) #EXACT FRAME THE U-TURN TAKES PLACE
turningpoints2 = np.array([])
for tu in range(0,len(data[:,0])-1):
    if incidences1[tu] * incidences1[tu+1] < 0:
        turningpoints1 = np.append(turningpoints1,tu)
    if incidences2[tu] *incidences2[tu+1] < 0:
        turningpoints2 = np.append(turningpoints2,tu)

#removing U-turns that fall within 100 of the beginning and the end
to_remove_1 = []
to_remove_2 = []

#Note - extra values removed at the end as due to the window size of 
#calculating influential neighbours, the last 1350 correlation values are ommitted
for l in range(len(turningpoints1)):
    if turningpoints1[l] < 100:
        to_remove_1.append(l)
    if abs(turningpoints1[l] - len(data[:,0])) < 1450:
        to_remove_1.append(l)
for k in range(len(turningpoints2)):
    if turningpoints2[k] < 100:
        to_remove_2.append(k)
    if abs(turningpoints2[k] - len(data[:,0])) < 1450:
        to_remove_2.append(k)

turningpoints1 = np.delete(turningpoints1,to_remove_1)
turningpoints2 = np.delete(turningpoints2,to_remove_2)

#Calculate number of U-turns    
tps = np.array([])
for t in range(0,len(turningpoints2)):
    tp = min(turningpoints1[t], turningpoints2[t])
    tps = np.append(tps, tp)

#Calculating start and ends of U-turns

#Start at 100 frames before to search for start of U-turn 
for pos in range(len(tps)):
    incst1 = incidences1[int(tps[pos])-100:int(tps[pos])]
    incst2 = incidences2[int(tps[pos])-100:int(tps[pos])]
    #Start at the frame of the U-turn to look for end of U-turn
    incen1 = incidences1[int(tps[pos]):int(tps[pos])+100]
    incen2 = incidences2[int(tps[pos]):int(tps[pos])+100]
    
    #Calculating start of each U-turn
    uturnstart1 = mov.moveav(incst1,5)
    uturnstart2 = mov.moveav(incst2,5)
    
    #Calculating end of each U-turn
    uturnend1 = mov.moveav(incen1,5)
    uturnend2 = mov.moveav(incen2,5)
    
    #Collective U-turn
    act_start = min(uturnstart1,uturnstart2)
    act_end = max(uturnend1,uturnend2)
    
    #Determining order of turning of fish and whether they are an influential neighbour
    thresh = 0.7 #Threshold for which an influential neighbour is defined
    if uturnstart1 < uturnstart2:
        if corr_values[int(tps[pos]-100+uturnstart1),0] > thresh:
            turning_rank_1 +=1
        elif corr_values[int(tps[pos]-100+uturnstart1),1] > thresh:
            turning_rank_2 +=1
    if uturnstart1 > uturnstart2:
        if corr_values[int(tps[pos]-100+uturnstart2),1] > thresh:
            turning_rank_1 +=1
        elif corr_values[int(tps[pos]-100+uturnstart2),0] >thresh:
            turning_rank_2 +=1
    
    #Position Rank
    
    #Group velocity calculation
    group_x,group_y = grp.groupvel(data[int(tps[pos]-100+act_start): \
                                        int(tps[pos]+act_end),6], \
                     data[int(tps[pos]-100+act_start): \
                          int(tps[pos]+act_end),7], \
                         data[int(tps[pos]-100+ act_start):\
                              int(tps[pos]+act_end),8], \
                             data[int(tps[pos]-100+act_start):\
                                  int(tps[pos]+act_end),9])
    proj1 = []
    proj2 = []
    
    #Calculating projection of each fish's velocity in direction of group centroid
    for frame in range(0,len(group_x)):
        grp_vel = np.array([group_x[frame],group_y[frame]])
        f1_vel = np.array([data[int(tps[pos]-100+act_start+frame),6], \
                           data[int(tps[pos]-100+act_start+frame),7]])
        f2_vel = np.array([data[int(tps[pos]-100+act_start+frame),8], \
                           data[int(tps[pos]-100+act_start+frame),9]])
        f1_dot = np.dot(f1_vel,grp_vel)
        f2_dot = np.dot(f2_vel,grp_vel)
        grp_mag = np.dot(grp_vel,grp_vel)
        
        pr1= float((f1_dot / grp_mag))
        pr2= float((f2_dot / grp_mag))
        proj1.append(pr1)
        proj2.append(pr2)
    
    #Determining order of fish at each frame and cross-referencing 
    #whether they are an influential neighbour using table of correlation values
    for loop in range(0,len(proj1)):
        if float(proj1[loop]) > float(proj2[loop]):
            if float(corr_values[int(tps[pos]-100+act_start +loop),0]) > thresh:
                pos_rank = np.append(pos_rank,1)
            elif float(corr_values[int(tps[pos]-100+act_start+loop),1]) > thresh:
                pos_rank = np.append(pos_rank,2)
        elif float(proj1[loop]) < float(proj2[loop]):
            if float(corr_values[int(tps[pos]-100+act_start+loop),1]) > thresh:
                pos_rank = np.append(pos_rank,1)
            elif float(corr_values[int(tps[pos]-100+act_start+loop),0]) > thresh:
                pos_rank = np.append(pos_rank,2)
        


#Turning Rank
proportion_rank1 = turning_rank_1/len(tps)
proportion_rank2 = turning_rank_2/len(tps)


ranks_turn = [proportion_rank1, proportion_rank2]
options = ['Position 1','Position 2']

#Plotting bar chart for Turning Rank of Infuential Neighbours 
plt.bar(options,ranks_turn)
plt.xlabel('Turning rank of Influential Neighbours')
plt.ylabel('Proportion')
plt.title('Plot showing Turning Rank of Influential Neighbours')
plt.show()

#Position Rank
first_pos = np.count_nonzero(pos_rank == 1)
sec_pos = np.count_nonzero(pos_rank == 2)

prop_first = first_pos/(len(pos_rank))
prop_sec = sec_pos/(len(pos_rank))

ranks_pos = [prop_first,prop_sec]

#PLotting bar chart for Position rank of Influentuial Neighbours
plt.bar(options, ranks_pos)
plt.xlabel('Position of Influential Neighbour')
plt.ylabel('Proportion')
plt.title('Position of Influential Neighbours during U-turns with 2-Fish')
plt.show()
        
            
print(turning_rank_1 + turning_rank_2)


        
        
        






