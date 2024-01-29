2D cellular automata majority rule

Cellular automata models are characterized by generating very complicated pattterns from extrmely simple rules. In CA, the evolution of a given site depends on its own state and on the neiughboring states. In this project we will emplement the majority rule in binary and 3-state models. For the binary case we will consider that the sites can be occupued by a bit ={0,1}, while in the 3-states automaton we will consider e.g., states {1,2,3}. This CA is governed by the majority rule. That is, each site will change to the most abundant state in its neighbourhood. For example, in the binary case, if the site constains 0 is surrounded by three 1's and one 0 then it will change to 1 in the next generation.

--------------------------

# 2D Cellular automata majority rule

## 1. Introduction

Cellular automata (CA) are numerical models that serve to replicate spatial dynamics. They use discrete time, variables and space in order to achieve this. They are mainly composed of three key components: cell state, neighborhood and rules. For the sake of simplicity, let us consider a one-dimentional grid of discretized space (for example, a line of pixels), we could let these pixels or cells be in two  different states, say 1 or 0. The latter would correspond to the cell state. Once the state is set, we can consider the neighborhood of a given cell. Once we have defined this neighborhood, which in our example would be the adjacent cells, we can introduce some rules. These rules determine how the state of each cell changes over time. They are typically based on the current state of a cell and the states of its neighbors. Even though at first sight this might seem like a simple idea, additional layers of complexity can be added in the form of higher spatial dimensions, more states considered, and including various rules 

- Capabilities to study a lot of problems within science and further

- In the present study, we will consider a two-dimensional cellular automaton with 3 states.


![Test](https://github.com/sofia-llacer-caro/cellular-automata/blob/main/cellular_automaton.gif)
