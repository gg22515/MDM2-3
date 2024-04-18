import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#epsilon = [3,5]

#Cmin = [0.995,0.99,0.95,0.5]
def influential(file,start_frame,num_frames):
    data_file = file
    start_row = start_frame
    df = pd.read_csv(data_file, skiprows=int(start_row),nrows=int(num_frames), names=['X1','Y1','H1','X2','Y2','H2','VX1','VY1','VX2','VY2','UVX1','UVY1','UVX2','UVY2'])

    invalid_strings = ['#DIV/0!', 'NA', '#VALUE!']
    for col in df.columns:
        for invalid_str in invalid_strings:
            df = df[df[col] != invalid_str]

    # Convert remaining values to floats
    df = df.apply(pd.to_numeric, errors='ignore')

    # Drop rows with NaN values
    df = df.dropna()

    fish1_uvx = df["UVX1"].to_numpy()
    fish1_uvy = df["UVY1"].to_numpy()
    fish2_uvx = df["UVX2"].to_numpy()
    fish2_uvy = df["UVY2"].to_numpy()

    fish1_velocities = np.stack((fish1_uvx, fish1_uvy), axis=1)  
    fish2_velocities = np.stack((fish2_uvx, fish2_uvy), axis=1)  

    w = 4  

    tau_0 = 30
    max_time_delay = 35  # Adjust this based on your analysis
    time_delays = range(tau_0, max_time_delay)  # Create your R_k

    Gamma_values = []
    for t_k in range(len(fish1_velocities) - w):  
        #max_correlation = 0.95  # Initialize to find the maximum
        correlation = 0.0
        for tau_r in range(min(t_k,max_time_delay)):
            
            # Calculate C_ij(t_k, tau_r, w) 
            start_index = t_k 
            end_index = t_k + w
            current_velocities = fish1_velocities[start_index:end_index]
            shifted_velocities = fish2_velocities[start_index - tau_r:end_index - tau_r]
            current_velocities = current_velocities.astype(float)  
            shifted_velocities = shifted_velocities.astype(float)

            dot_products = np.einsum('ij,ij->i', current_velocities, shifted_velocities)
            correlation = np.average(dot_products)  # Average correlation within window

            #max_correlation = max(max_correlation, correlation)
            #print(correlation)
            
        Gamma_values.append(correlation)        

        


    #for t_k, gamma in enumerate(Gamma_values):
    #    if gamma > 0.95:  # Adjust the threshold as needed
            print(f"Fish 2 is likely influencing Fish 1 at time t_k = {t_k}") 
    #    else: 
            print(f"Fish 1 is likely influencing Fish 2 at time t_k = {t_k}") 
    return Gamma_values[1:] #skips the first value as this is just whatever is set as correlation above so total size will be + -1 + -w*n eg 1000 rows will give you 979 or 800 rows will give you 779 depends on how many loops are ran


 
#put file to test function
#data_file = "C:\\Users\\TheBuild\\OneDrive - University of Bristol\\Documents\\University\\Year 2\\MDM2\\Project3\\Python Code\\MDM2-3\\FishData\\exp02H20141128_16h06updated.csv"

#gamma = influential(data_file,1,800)
##number_of_elements = len(gamma)
##print(f"The size of the gamma list is: {number_of_elements}")
#print(gamma)