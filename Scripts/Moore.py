# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 20:37:24 2024

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
steps = 30

# States: uncomment the desired one
states=[0,1]
#states = [1,2,3]
#states = list(range(20))

##############################


### Functions ################
# Function for the majority rule without the current cell
def most_frequent_numbers_wo_center(matrix, row, column):
    # Define a 3x3 mask centered at the position (row, column)
    mask = matrix[max(0, row-1):row+2, max(0, column-1):column+2].copy()
    mask[1, 1] = -1

    # Flatten the mask to count the frequency of each number
    values, frequencies = np.unique(mask, return_counts=True)

    # Find the maximum frequency value
    max_frequency = np.max(frequencies)

    # Find all numbers that have the maximum frequency
    most_frequent_numbers = values[frequencies == max_frequency]

    return most_frequent_numbers.tolist()

# Function for the majority rule with the current cell
def most_frequent_numbers_w_center(matrix, row, column):
    # Define a 3x3 mask centered at the position (row, column)
    mask = matrix[max(0, row-1):row+2, max(0, column-1):column+2]

    # Flatten the mask to count the frequency of each number
    values, frequencies = np.unique(mask, return_counts=True)

    # Find the maximum frequency value
    max_frequency = np.max(frequencies)

    # Find all numbers that have the maximum frequency
    most_frequent_numbers = values[frequencies == max_frequency]

    return most_frequent_numbers.tolist()

# Update function for the matrix
def update(frameNum, img, grid, text, rows, cols):
    if frameNum == 0:
        # On the first frame, display the initial state
        img.set_data(grid)
        text.set_text(f'Step: {frameNum}/{steps}')
        new_grid = grid.copy()
        
    else:
        new_grid = grid.copy()
        for i in range(rows):
            for j in range(cols):
                # Get the most frequent numbers without center
                result_no_center = most_frequent_numbers_wo_center(grid, i, j)
                
                if len(result_no_center) > 1:
                    # If there are multiple most frequent numbers without center,
                    # get the most frequent numbers with center
                    result_with_center = most_frequent_numbers_w_center(grid, i, j)
                    
                    if len(result_with_center) > 1:
                        # If there are still multiple most frequent numbers with center,
                        # randomly choose one
                        new_grid[i, j] = random.choice(result_with_center)
                    else:
                        # If there is only one most frequent number with center, use it
                        new_grid[i, j] = result_with_center[0]
                else:
                    # If there is only one most frequent number without center, use it
                    new_grid[i, j] = result_no_center[0]
    
        img.set_data(new_grid)
        grid[:] = new_grid
        text.set_text(f'Step: {frameNum}/{steps}')
    
    return img, text

##############################


### Execution of model #######
# Initialization of the initial state randomly
initial_state = np.random.choice(states, size=(rows, cols))

# Display configuration and animation
fig, ax = plt.subplots()
cmap = plt.get_cmap('viridis', len(states))
img = ax.imshow(initial_state, cmap=cmap, interpolation='nearest')
text = ax.text(0.5, -0.1, '', transform=ax.transAxes, ha='center', va='center')

ani = animation.FuncAnimation(fig, update, fargs=(img, initial_state, text, rows, cols),
                              frames=steps+1, interval=500, save_count=steps)

ani.save('Cellular_Automata_Moore.gif', writer='imagemagick', fps=2)
plt.show()



