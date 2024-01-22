# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 14:21:05 2023

@author: Roberto
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Parameters
rows, cols = 100, 100
steps = 50

# Random initialization of the initial state
initial_state = np.random.choice([1, 2, 3], size=(rows, cols))

def most_frequent_numbers_wo_center(matrix, row, col):
    # Define the 3x3 mask centered at position (row, col)
    mask = matrix[max(0, row-1):row+2, max(0, col-1):col+2].copy()
    mask[1, 1] = 0

    # Flatten the mask to count the frequency of each number
    values, frequencies = np.unique(mask, return_counts=True)

    # Find the maximum frequency value
    max_frequency = np.max(frequencies)

    # Find all numbers that have the maximum frequency
    most_frequent_numbers = values[frequencies == max_frequency]

    return most_frequent_numbers.tolist()

def most_frequent_numbers_w_center(matrix, row, col):
    # Define the 3x3 mask centered at position (row, col)
    mask = matrix[max(0, row-1):row+2, max(0, col-1):col+2]
    # Flatten the mask to count the frequency of each number
    values, frequencies = np.unique(mask, return_counts=True)

    # Find the maximum frequency value
    max_frequency = np.max(frequencies)

    # Find all numbers that have the maximum frequency
    most_frequent_numbers = values[frequencies == max_frequency]

    return most_frequent_numbers.tolist()

# Function to apply the rules and evolve the cellular automaton
def update(frameNum, img, grid, text, rows, cols):
    if frameNum == 0:
        # In the first frame, display the initial state
        img.set_data(grid)
        text.set_text(f'Step: {frameNum}/{steps}')
        new_grid = grid.copy()
        
    else:
        new_grid = grid.copy()
        for i in range(rows):
            for j in range(cols):
                # Apply your rules here (example: rule of sum modulo 2)
                result = most_frequent_numbers_wo_center(grid, i, j)
                if len(result) > 1:
                    result = most_frequent_numbers_w_center(grid, i, j)
                    if len(result) > 1:
                        new_grid[i, j] = random.choice(result)
                    else:
                        new_grid[i, j] = result[0]
                else:
                    new_grid[i, j] = result[0]  # Adjust the rule according to your needs
    
        img.set_data(new_grid)
        grid[:] = new_grid
        
        text.set_text(f'Step: {frameNum}/{steps}')
    
    return img, text

# Visualization setup
fig, ax = plt.subplots()
cmap = plt.get_cmap('viridis', 3)
img = ax.imshow(initial_state, cmap=cmap, interpolation='nearest')

text = ax.text(0.5, -0.1, '', transform=ax.transAxes, ha='center', va='center')
ani = animation.FuncAnimation(fig, update, fargs=(img, initial_state, text, rows, cols),
                              frames=steps+1, interval=500, save_count=steps)

ani.save('cellular_automaton.gif', writer='imagemagick', fps=2)
plt.show()
