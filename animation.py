import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

data_file = "C:/Users/TheBuild/OneDrive - University of Bristol/Documents/University/Year 2/MDM2/Project3/FishData/exp02H20141128_16h06updated.csv"
#change this to pick which row to start and change nrows to say how many rows to animate
start_row = 150
df = pd.read_csv(data_file,skiprows=start_row, nrows=500, names=['X1','Y1','H1','X2','Y2','H2','VX1','VY1','VX2','VY2','UVX1','UVY1','UVX2','UVY2'])

fig, ax = plt.subplots()
line1, = ax.plot([], [], label='Fish 1') 
line2, = ax.plot([], [], label='Fish 2') 

ax.set_xlim(-400, 400)
ax.set_ylim(-400, 400)

frame_text = ax.text(0.05, 0.95, '', ha='left', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))


def animate(frame_number):
    fish1_X = df['X1'][:frame_number + 1]
    fish1_Y = df['Y1'][:frame_number + 1]
    fish2_X = df['X2'][:frame_number + 1]
    fish2_Y = df['Y2'][:frame_number + 1]

    line1.set_data(fish1_X, fish1_Y)
    line2.set_data(fish2_X, fish2_Y)
    frame_text.set_text(f'Frame: {frame_number + 1}')

    return line1, line2, frame_text
ani = animation.FuncAnimation(
    fig, animate, frames=len(df), interval=20, blit=True
)


ax.set_xlabel('X-Position')
ax.set_ylabel('Y-Position')
ax.set_title('Animated Movement of Two Fish')
ax.legend()
plt.show()