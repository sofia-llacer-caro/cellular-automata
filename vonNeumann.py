# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:31:58 2024

@author: Roberto
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Parámetros
rows, cols = 100, 100
steps = 30
states=[0,1]
# states = [1,2,3]
states = list(range(20))

cross = np.array([[np.nan, 1, np.nan],
                  [1, 1, 1],
                  [np.nan, 1, np.nan]])

cross_wo = np.array([[np.nan, 1, np.nan],
                  [1, np.nan, 1],
                  [np.nan, 1, np.nan]])



# Inicialización del estado inicial de manera aleatoria
initial_state = np.random.choice(states, size=(rows, cols))
padded_matrix=np.full((initial_state.shape[0] + 2, initial_state.shape[1] + 2), np.nan)
padded_matrix[1:-1, 1:-1] = initial_state

def numeros_mas_frecuentes_w_center_cross(matriz, fila, columna):
    f=fila+1
    c=columna+1
    mascara = matriz[f-1:f+2, c-1:c+2]*cross
    mascara = mascara[~np.isnan(mascara)]

    # Aplana la máscara para contar la frecuencia de cada número
    valores, frecuencias = np.unique(mascara, return_counts=True)

    # Encuentra el valor máximo de frecuencia
    max_frecuencia = np.max(frecuencias)

    # Encuentra todos los números que tienen la frecuencia máxima
    numeros_mas_frecuentes = valores[frecuencias == max_frecuencia]
    
    return numeros_mas_frecuentes.tolist()


def numeros_mas_frecuentes_wo_center_cross(matriz, fila, columna):
    f=fila+1
    c=columna+1
    mascara = matriz[f-1:f+2, c-1:c+2]*cross_wo
    mascara = mascara[~np.isnan(mascara)]

    # Aplana la máscara para contar la frecuencia de cada número
    valores, frecuencias = np.unique(mascara, return_counts=True)

    # Encuentra el valor máximo de frecuencia
    max_frecuencia = np.max(frecuencias)

    # Encuentra todos los números que tienen la frecuencia máxima
    numeros_mas_frecuentes = valores[frecuencias == max_frecuencia]

    return numeros_mas_frecuentes.tolist()

# Función para aplicar las reglas y evolucionar el automata
def update(frameNum, img, grid, padded_grid, text, rows, cols):
    if frameNum == 0:
        # En el primer frame, mostrar el estado inicial
        img.set_data(grid)
        text.set_text(f'Step: {frameNum}/{steps}')
        new_grid = grid.copy()
        
    else:
        new_grid = grid.copy()
        for i in range(rows):
            for j in range(cols):
                resultado = numeros_mas_frecuentes_wo_center_cross(padded_grid, i, j)
                if len(resultado)>1:
                    resultado = numeros_mas_frecuentes_w_center_cross(padded_grid, i, j)
                    if len(resultado)>1:
                        new_grid[i, j] = random.choice(resultado)
                    else:
                        new_grid[i, j] = resultado[0]
                else:
                    new_grid[i, j] = resultado[0]  
                    
        img.set_data(new_grid)
        grid[:] = new_grid
        padded_matrix[1:-1, 1:-1] = new_grid
        
        text.set_text(f'Step: {frameNum}/{steps}')
    
    return img, text

# Configuración de la visualización
fig, ax = plt.subplots()
cmap = plt.get_cmap('viridis', len(states))
img = ax.imshow(initial_state, cmap=cmap, interpolation='nearest')

text = ax.text(0.5, -0.1, '', transform=ax.transAxes, ha='center', va='center')
ani = animation.FuncAnimation(fig, update, fargs=(img, initial_state, padded_matrix, text, rows, cols),
                              frames=steps+1, interval=500, save_count=steps)

valores, frecuencias = np.unique(initial_state, return_counts=True)

ani.save('automata_celular_simpler_cruz.gif', writer='imagemagick', fps=2)
plt.show()




