# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:53:08 2024

@author: Hassan Miah
"""

import pandas as pd
import numpy as np
import polarisation2 as p2
import matplotlib.pyplot as plt
import incidence as inc
import turnangle as tna
import moveav as mov
import FunctionInfluentialFive as inf
import Fivegroupvelocity as grp

#Calling file with correlation values for all pairwise combinations of fish
corr_file = "Corr_values5fish.csv"
myDF = pd.read_csv(corr_file)
myDF.replace('N/A',0,inplace=True)
myDF.fillna(0,inplace=True)
corr_values = myDF.values

#Intialising lists for turning and position ranks for later
turn_rank = []
pos_rank = []

#Obtaining data for fish
file = "5fishappended\exp05H20140926_10h50.csv"
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
myDF = df.dropna()
df = df.astype(float)
data = df.values

#Initialising angle of incidence arrays for each fish
incidences1 = np.array([])
incidences2 = np.array([])
incidences3 = np.array([])
incidences4 = np.array([])
incidences5 = np.array([])

#Calculating angles for incidences over entire dataset
for val in range(0,len(data[:,0])):
    angle1 = inc.incidence(data[val,0],data[val,1],data[val,2])
    angle2 = inc.incidence(data[val,3],data[val,4],data[val,5])
    angle3 = inc.incidence(data[val,6],data[val,7],data[val,8])
    angle4 = inc.incidence(data[val,9],data[val,10],data[val,11])
    angle5 = inc.incidence(data[val,12],data[val,13],data[val,14])
    
    incidences1 = np.append(incidences1,angle1)
    incidences2 = np.append(incidences2,angle2)
    incidences3 = np.append(incidences3,angle3)
    incidences4 = np.append(incidences4,angle4)
    incidences5 = np.append(incidences5,angle5)
 
#All incidences must be in range [-pi,pi]
incidences1 = np.mod(incidences1 + np.pi, 2 * np.pi) - np.pi
incidences2 = np.mod(incidences2 + np.pi, 2 * np.pi) - np.pi
incidences3 = np.mod(incidences3 + np.pi, 2 * np.pi) - np.pi
incidences4 = np.mod(incidences4 + np.pi, 2 * np.pi) - np.pi
incidences5 = np.mod(incidences5 + np.pi, 2 * np.pi) - np.pi

turningpoints1 = np.array([])
turningpoints2 = np.array([])
turningpoints3 = np.array([])
turningpoints4 = np.array([])
turningpoints5 = np.array([])

#Calculating turning points by looking for changes in sign
for tu in range(0,len(data[:,0])-1):
    if incidences1[tu] * incidences1[tu+1] < 0:
        turningpoints1 = np.append(turningpoints1,tu)
    if incidences2[tu] *incidences2[tu+1] < 0:
        turningpoints2 = np.append(turningpoints2,tu)
    if incidences3[tu] * incidences3[tu+1] < 0:
        turningpoints3 = np.append(turningpoints3,tu)
    if incidences4[tu] *incidences4[tu+1] < 0:
        turningpoints4 = np.append(turningpoints4,tu)
    if incidences5[tu] * incidences5[tu+1] < 0:
        turningpoints5 = np.append(turningpoints5,tu)

#Calculate number of U-turns    
num_tps = min(len(turningpoints1),len(turningpoints2),len(turningpoints3), \
              len(turningpoints4),len(turningpoints5))

tps = np.array([])
for t in range(0,num_tps):
    tp = min(turningpoints1[t], turningpoints2[t],turningpoints3[t] , \
             turningpoints4[t], turningpoints5[t])
    tps = np.append(tps,tp)

#removing U-turns that fall within 100 of the beginning and the end
to_remove = []
for l in range(0,len(tps)):
    if tps[l] < 100:
        to_remove.append(l)
    if abs(tps[l] - len(data[:,0])) < 112: 
        to_remove.append(l)
#Note - extra 12 values removed at the end as due to the window size of 
#calcualting influential neighbours, the last 12 correlation values are ommitted
tps = np.delete(tps,to_remove)

#Calculating start and ends of U-turns

#Start at 100 frames before to search for start of U-turn 
for pos in range(0,len(tps)):
    incst1 = incidences1[int(tps[pos])-100:int(tps[pos])]
    incst2 = incidences2[int(tps[pos])-100:int(tps[pos])]
    incst3 = incidences3[int(tps[pos])-100:int(tps[pos])]
    incst4 = incidences4[int(tps[pos])-100:int(tps[pos])]
    incst5 = incidences5[int(tps[pos])-100:int(tps[pos])]
#Start at the frame of the U-turn to look for end of U-turn
    incen1 = incidences1[int(tps[pos]):int(tps[pos])+100]
    incen2 = incidences2[int(tps[pos]):int(tps[pos])+100]
    incen3 = incidences3[int(tps[pos]):int(tps[pos])+100]
    incen4 = incidences4[int(tps[pos]):int(tps[pos])+100]
    incen5 = incidences5[int(tps[pos]):int(tps[pos])+100]
    
    #Turning Rank 
    
    #Calculating start of each U-turn
    uturnstart1 = mov.moveav(incst1,5)
    uturnstart2 = mov.moveav(incst2,5)
    uturnstart3 = mov.moveav(incst3,5)
    uturnstart4 = mov.moveav(incst4,5)
    uturnstart5 = mov.moveav(incst5,5)
    #Calculating end of each U-turn
    uturnend1 = mov.moveav(incen1,5)
    uturnend2 = mov.moveav(incen2,5)
    uturnend3 = mov.moveav(incen3,5)
    uturnend4 = mov.moveav(incen4,5)
    uturnend5 = mov.moveav(incen5,5)
    
    
    uturn_dict = {'f1':uturnstart1,'f2':uturnstart2,'f3':uturnstart3, \
                  'f4':uturnstart4,'f5':uturnstart5}
    fish = list(uturn_dict.keys())
    uturnstarts = list(uturn_dict.values())
    uturnends = [uturnend1,uturnend2,uturnend3,uturnend4,uturnend5]
    
    #Ordering the fish in terms of which fish turns first using value of start
    sorted_uturn = dict(sorted(uturn_dict.items(), key=lambda item:item[1]))
    thresh = 0.95 #Threshold for which an influential neighbour is defined
    #Columns in the spreadsheet of correlated values
    columns = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19]]
    #cross checking the positions of fish and whether they influence other fish
    for col in range(0,5):
        vals_to_check = corr_values[uturnstarts[col],columns[col]]
        is_inf = any(vals_to_check > thresh)
        if is_inf:
            inf_fish = fish[col]
            index = list(sorted_uturn.keys()).index(inf_fish)
            turn_rank.append(index+1)
            
    #Position Rank
    act_start = min(uturnstarts) #start of collective U-turn
    act_end = max(uturnends) #End of collective U-turn
    
    #Group velocity
    group_x,group_y = grp.groupvel(data[int(tps[pos]-100+act_start):\
                                        int(tps[pos]+act_end)])
    
     #Calculating projections of fish's velocity in direction of group centroid  
    for frame in range(0,len(group_x)):
        grp_vel = np.array([group_x[frame],group_y[frame]])
        grp_mag = np.dot(grp_vel,grp_vel)
        f1_dot = np.dot(np.array([data[int(tps[pos]-100+act_start+frame),15],\
                                  data[int(tps[pos]-100+act_start+frame),16]]),grp_vel)
        f2_dot = np.dot(np.array([data[int(tps[pos]-100+act_start+frame),19],\
                                  data[int(tps[pos]-100+act_start+frame),20]]),grp_vel)
        f3_dot = np.dot(np.array([data[int(tps[pos]-100+act_start+frame),23],\
                                  data[int(tps[pos]-100+act_start+frame),24]]),grp_vel)
        f4_dot = np.dot(np.array([data[int(tps[pos]-100+act_start+frame),27],\
                                  data[int(tps[pos]-100+act_start+frame),28]]),grp_vel)
        f5_dot = np.dot(np.array([data[int(tps[pos]-100+act_start+frame),31],\
                                  data[int(tps[pos]-100+act_start+frame),32]]),grp_vel)
        pr1 = float(f1_dot/grp_mag)
        pr2 = float(f2_dot/grp_mag)
        pr3 = float(f3_dot/grp_mag)
        pr4 = float(f4_dot/grp_mag)
        pr5 = float(f5_dot/grp_mag)
        
        
        projections = {'f1':pr1,'f2':pr2,'f3':pr3,'f4':pr4,'f5':pr5}
        
        #Sort projections in descending order of who is the most ahead
        sorted_projs = sorted_uturn = dict(sorted(projections.items(),\
                                                  key=lambda item:item[1], \
                                                      reverse=True))
        #Cross checking whether the fish in each position is an influential fish
        for col in range(0,5):
            checking = corr_values[int(tps[pos]-100+act_start+frame),columns[col]]
            inf_it_is = any(checking > thresh)
            if inf_it_is:
                influent_fish = fish[col]
                indexing = list(sorted_projs.keys()).index(influent_fish)
                pos_rank.append(index+1)
                
#Counting the number of influential neighbours in each position (turning rank)            
turn1 = turn_rank.count(1)
turn2 = turn_rank.count(2)
turn3 = turn_rank.count(3)
turn4 = turn_rank.count(4)
turn5 = turn_rank.count(5)

turn_ranks = [turn1,turn2,turn3,turn4,turn5]
total_infs = sum(turn_ranks)

#Converting to proportions
prop_turns = [val/total_infs for val in turn_ranks]

options = ['Position 1', 'Position 2', 'Position 3', 'Position 4', \
           'Position 5']
colors = ['red','blue','green','purple','orange']

#Plotting bar chart for turning rank
plt.bar(options,prop_turns,color = colors)
plt.xlabel('Turning Rank of Influential Neighbours')
plt.ylabel('Proportion')
plt.title('Plot showing Turning Ranks of Influential Neighbours')
plt.show()

#Counting number of influential neighbours in each position (position rank)
first_pos = pos_rank.count(1)
sec_pos = pos_rank.count(2)
three_pos = pos_rank.count(3)
four_pos = pos_rank.count(4)
five_pos = pos_rank.count(5)

pos_ranks = [first_pos,sec_pos,three_pos,four_pos,five_pos]

sum_infs = sum(pos_ranks)
#Converting to proportions
prop_positions = [item/sum_infs for item in pos_ranks]
pop_positions = prop_positions.reverse()

#Plotting bar chart for position rank
plt.bar(options,prop_positions,color=colors)
plt.xlabel('Position Rank of Influential Neighbours')
plt.ylabel('Proportion')
plt.title('Position of Influential Neighbours during U-turns with 5-Fish')
plt.show()
        
        

            

        

    
    
    
    
    
