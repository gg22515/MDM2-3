README:
This README file includes descriptions and functions of all relevant code used for this project and how to run them. ALL codes may be ran in relevant software without any extra steps. [PLEASE CORRECT THIS IF CODES ARE USED WHICH ARE OTHERWISE]


2fishPosTurn: Analyses all fish U-turns that occur in a single dataset and outputs two bar charts, one for the Position Rank of fish that are influential neighbours to any other fish, measured at every frame of U-turn, and the second bar chart showing a measure of the order of which fish start their U-turn first, and whether or not they are also an influential neighbour

5fishPosTurn: Exactly the same reasoning as 2fishPosTurn, except this process is repeated for Five Fish.

Distance Ranking: Ranks the distances of a fish from the leader fish, based on Euclidean distance at each frame during a U-turn. The code produces a bar chart which defines the proportion of the time a fish spends during a U-turn being (N-1) fish behind the leader fish (defined as the fish at the front of the group). This analysis only holds for five fish, as this approach is null when there are only 2 fish.

incidence: Calculates the angle between the fish and the wall of the petri-dish, measured in Radians. Outputs an array containing the incidence for a single fish at each frame 

threshold_value: Function which calculates a threshold value for which the first time the incidence value exceeds this, the corresponding frame where this occurs represents the start or end of the U-turn, depending on analysis. This is completed by first computing the moving average of a specified set of frames.

FunctionInfluential & FunctionInfluentialFive: These functions are used to compute the correlation values between Fish_i and a Fish_j with a given time delay. 'FunctionInfluential' is used for 2-fish data, and 'FunctionInfluentialFive' is used with 5-fish data.
This requires at least 39 frames of data in its current state but can be modified depending on your analysis. Window size will change the amount of frames looked at for correlation analysis, currently set to 4. Tau_r is the time delay, this is to account for potential delays in influence. 
It helps to compare the current velocties of one fish in the current window with the shifted velocties of the other fish for different time delays. Within each window and for each time delaym the average dot product is calculated resulting in a single correlation value.
This value represents the strength and direction of the linear relationship between the velocities of the two fish for that specific time window and time delay. The difference between the two functions is that one was built for two fish and the other for five. 
The code is layed out to take in raw data and appened the correlation values of each fish pair to a new excel sheet. If you want to find unique pairs on their own you can modify the code and use the pairing you want to observe in the 4th argument. 
This would be in the format ' "2to1" ' this would give back the influence of fish 2 over fish 1. 

fivegroupvelocity & twogroupvelocity: Functions which are utilised to compute a group centroid based on the velocities of the all of the fish at a given frame. 







