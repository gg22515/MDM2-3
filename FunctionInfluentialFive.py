import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def influential(file,start_frame,num_frames,direction):
    data_file = file
    start_row = start_frame
    df = pd.read_csv(data_file, skiprows=int(start_row),nrows=int(num_frames), names=['X1','Y1','H1','X2','Y2','H2', 'X3', 'Y3', 'H3', 'X4', 'Y4', 'H4', 'X5', 'Y5', 'H5',
                            'VX1','VY1','VX2','VY2', 'VX3', 'VY3', 'VX4', 'VY4', 'VX5', 'VY5',
                            'UX1','UY1','UX2','UY2', 'UX3', 'UY3', 'UX4', 'UY4', 'UX5', 'UY5'])

    invalid_strings = ['#DIV/0!', 'NA', '#VALUE!']
    for col in df.columns:
        for invalid_str in invalid_strings:
            df = df[df[col] != invalid_str]

    # Convert remaining values to floats
    df = df.apply(pd.to_numeric, errors='ignore')

    # Drop rows with NaN values
    df = df.dropna()

    fish_vx = [df[f"VX{i}"].to_numpy() for i in range(1, 6)]
    fish_vy = [df[f"VY{i}"].to_numpy() for i in range(1, 6)]

    fish_uvx = [fish_vx[i]/(np.sqrt(fish_vx[i]**2+fish_vy[i]**2)) for i in range(5)]
    fish_uvy = [fish_vy[i]/(np.sqrt(fish_vx[i]**2+fish_vy[i]**2)) for i in range(5)]

    fish_velocities = [np.stack((uvx, uvy), axis=1) for uvx, uvy in zip(fish_uvx, fish_uvy)]

    w = 4  
    tau_0 = 30
    max_time_delay = 35  
    #time_delays = range(tau_0, max_time_delay)  # Create your R_k

    Gamma_values = []

    for i in range(5):
        for j in range(i+1, 5):
            if direction == f"{i+1}to{j+1}":
                for t_k in range(len(fish_velocities[i]) - w):  
                    correlation = 0.0
                    for tau_r in range(min(t_k,max_time_delay)):
                        start_index = t_k 
                        end_index = t_k + w
                        current_velocities = fish_velocities[i][start_index:end_index]
                        shifted_velocities = fish_velocities[j][start_index - tau_r:end_index - tau_r]
                        current_velocities = current_velocities.astype(float)  
                        shifted_velocities = shifted_velocities.astype(float)

                        dot_products = np.einsum('ij,ij->i', current_velocities, shifted_velocities)
                        correlation = np.average(dot_products)  # Average correlation within window
                
                    Gamma_values.append(correlation)   
            elif direction == f"{j+1}to{i+1}":
                for t_k in range(len(fish_velocities[j]) - w):  
                    correlation = 0.0
                    for tau_r in range(min(t_k,max_time_delay)):
                        start_index = t_k 
                        end_index = t_k + w
                        current_velocities = fish_velocities[j][start_index:end_index]
                        shifted_velocities = fish_velocities[i][start_index - tau_r:end_index - tau_r]
                        current_velocities = current_velocities.astype(float)  
                        shifted_velocities = shifted_velocities.astype(float)

                        dot_products = np.einsum('ij,ij->i', current_velocities, shifted_velocities)
                        correlation = np.average(dot_products)  # Average correlation within window
                
                    Gamma_values.append(correlation)    

    return Gamma_values[1:]




data_file = "C:\\Users\\TheBuild\\OneDrive - University of Bristol\\Documents\\University\\Year 2\\MDM2\\Project3\\Python Code\\MDM2-3\\FishData\\5fishappended\\exp05H20140926_10h50.csv"
gamma = influential(data_file, 10000, 800,"3to2")  # looking at uturn of 5 fish 

#print(len(gamma))
print("3to2")
for i, gamma in enumerate(gamma):
    if gamma > 0.95:
        print(f"Gamma value over 0.95 found at frame number {i+1}, value: {gamma}")

#gamma = influential(data_file, 10000, 800,"2to3")  

#print("2to3")

#for i, gamma in enumerate(gamma):
#    if gamma > 0.95:
 #       print(f"Gamma value over 0.95 found at frame number {i+1}, value: {gamma}")
