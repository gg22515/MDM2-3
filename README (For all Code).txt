README:
This README file includes descriptions and functions of all relevant code used for this project and how to run them.


2fishPosTurn: Analyses all fish U-turns that occur in a single dataset and outputs two bar charts, one for the Position Rank of fish that are influential neighbours to any other fish, measured at every frame of U-turn, and the second bar chart showing a measure of the order of which fish start their U-turn first, and whether or not they are also an influential neighbour

5fishPosTurn: Exactly the same reasoning as 2fishPosTurn, except this process is repeated for Five Fish.

Distance Ranking: Ranks the distances of a fish from the leader fish, based on Euclidean distance at each frame during a U-turn. The code produces a bar chart which defines the proportion of the time a fish spends during a U-turn being (N-1) fish behind the leader fish (defined as the fish at the front of the group). This analysis only holds for five fish, as this approach is null when there are only 2 fish.

incidence: Calculates the angle between the fish and the wall of the petri-dish, measured in Radians. Outputs an array containing the incidence for a single fish at each frame 

threshold_value: Function which calculates a threshold value for which the first time the incidence value exceeds this, the corresponding frame where this occurs represents the start or end of the U-turn, depending on analysis. This is completed by first computing the moving average of a specified set of frames.

FunctionInfluential & FunctionInfluentialFive: These functions are used to compute the correlation values between Fish_i and a Fish_j with a given time delay. 'FunctionInfluential' is used for 2-fish data, and 'FunctionInfluentialFive' is used with 5-fish data.

fivegroupvelocity & twogroupvelocity: Functions which are utilised to compute a group centroid based on the velocities of the all of the fish at a given frame. 







