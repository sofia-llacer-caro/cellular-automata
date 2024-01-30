# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:31:58 2024

"""

### Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
##############################


### Parmeters of the model ###
# Size and iterations
rows, cols = 150, 150
steps = 15

# States: uncomment the desired one
states=[0,1]
#states = [1,2,3]
#states = list(range(20))

##############################


### Used masks ###############
cross = np.array([[np.nan, 1, np.nan],
                  [1, 1, 1],
                  [np.nan, 1, np.nan]])

cross_wo = np.array([[np.nan, 1, np.nan],
                  [1, np.nan, 1],
                  [np.nan, 1, np.nan]])

##############################

### Functions ################
# Function for the majority rule without the current cell
def most_frequent_numbers_w_center_cross(matrix, row, column):
    f = row + 1
    c = column + 1
    mask = matrix[f-1:f+2, c-1:c+2] * cross
    mask = mask[~np.isnan(mask)]

    # Flatten the mask to count the frequency of each number
    values, frequencies = np.unique(mask, return_counts=True)

    # Find the maximum frequency value
    max_frequency = np.max(frequencies)

    # Find all numbers that have the maximum frequency
    most_frequent_numbers = values[frequencies == max_frequency]

    return most_frequent_numbers.tolist()

# Function for the majority rule with the current cell
def most_frequent_numbers_wo_center_cross(matrix, row, column):
    f = row + 1
    c = column + 1
    mask = matrix[f-1:f+2, c-1:c+2] * cross_wo
    mask = mask[~np.isnan(mask)]

    # Flatten the mask to count the frequency of each number
    values, frequencies = np.unique(mask, return_counts=True)

    # Find the maximum frequency value
    max_frequency = np.max(frequencies)

    # Find all numbers that have the maximum frequency
    most_frequent_numbers = values[frequencies == max_frequency]

    return most_frequent_numbers.tolist()

# Update function for the matrix
def update(frameNum, img, grid, padded_grid, text, rows, cols):
    if frameNum == 0:
        # On the first frame, display the initial state
        img.set_data(grid)
        text.set_text(f'Step: {frameNum}/{steps}')
        new_grid = grid.copy()
        
    else:
        new_grid = grid.copy()
        for i in range(rows):
            for j in range(cols):
                result_no_center = most_frequent_numbers_wo_center_cross(padded_grid, i, j)
                
                if len(result_no_center) > 1:
                    result_with_center = most_frequent_numbers_w_center_cross(padded_grid, i, j)
                    
                    if len(result_with_center) > 1:
                        new_grid[i, j] = random.choice(result_with_center)
                    else:
                        new_grid[i, j] = result_with_center[0]
                else:
                    new_grid[i, j] = result_no_center[0]  
                    
        img.set_data(new_grid)
        grid[:] = new_grid
        padded_grid[1:-1, 1:-1] = new_grid
        
        text.set_text(f'Step: {frameNum}/{steps}')
    
    return img, text

##############################


### Execution of model #######
# Initialization of the initial state randomly and pad adding
initial_state = np.random.choice(states, size=(rows, cols))
padded_matrix=np.full((initial_state.shape[0] + 2, initial_state.shape[1] + 2), np.nan)
padded_matrix[1:-1, 1:-1] = initial_state


# Display configuration and animation
fig, ax = plt.subplots()
cmap = plt.get_cmap('viridis', len(states))
img = ax.imshow(initial_state, cmap=cmap, interpolation='nearest')

text = ax.text(0.5, -0.1, '', transform=ax.transAxes, ha='center', va='center')
ani = animation.FuncAnimation(fig, update, fargs=(img, initial_state, padded_matrix, text, rows, cols),
                              frames=steps+1, interval=500, save_count=steps)

valores, frecuencias = np.unique(initial_state, return_counts=True)

ani.save('Cellular_Automata_vonNeumann.gif', writer='imagemagick', fps=2)
plt.show()




