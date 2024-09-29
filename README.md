<h1 align="center">
  
  ![zebra](https://github.com/user-attachments/assets/44729d07-9ec8-420c-852e-5d9910014bd1)

  
Cellular Automata 

  <br>
</h1>

<h4 align="center"> A project for Computational Biology course, Bar-Ilan University.


<p align="center">
  <a href="#description">Description</a> •
  <a href="#cellular-automata-rules">Cellular Automata Rules</a> •
  <a href="#the-scale">The Scale</a> •
  <a href="#difference-between-runs">Difference Between Runs</a> •
  <a href="#dependencies">Dependencies</a> •
    <a href="#installing-and-executing">Installing And Executing</a> •
  <a href="#author">Author</a> 
</p>

## Description

This project implements a cellular automaton simulation based on [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life). The simulation explores the emergence of complex patterns and behaviors from simple rules, with a specific focus on generating zebra-like patterns. This demonstrates principles relevant to computational biology, complex systems, and pattern formation in nature.

### Key features of the simulation:

**Cyclic Grid**: The simulation runs on a cyclic grid, meaning that the edges of the grid wrap around. This creates a toroidal topology where cells on the right edge are neighbors with cells on the left edge, and cells on the top edge are neighbors with cells on the bottom edge.

**Game of Life Mechanics**: Each cell in the grid can be in one of two states: alive or dead. In each generation, the state of each cell is updated based on its current state and the states of its eight neighboring cells (including diagonal neighbors).

**Zebra Pattern Goal**: Unlike the traditional Game of Life, which often results in chaotic or stable patterns, this simulation aims to produce zebra-like striped patterns. This is achieved through careful tuning of the initial conditions and rule set.

**Iterative Process**: The simulation progresses through multiple generations. In each generation, the rules are applied simultaneously to all cells in the grid, creating a new generation based on the previous one.

**Emergent Behavior**: Despite the simplicity of the rules, complex patterns emerge over time. The zebra patterns that form demonstrate how simple, local interactions can lead to global, organized structures - a principle often observed in biological systems.

This project not only showcases the power of cellular automata in modeling complex systems but also provides insights into how natural patterns, like zebra stripes, might emerge from simple underlying rules.
  
## Cellular Automata Rules
  
The cellular automaton follows the Game of Life rules:

![111](https://github.com/user-attachments/assets/ebf2f398-a7d1-4ffb-969c-44bad940c2e3)


These rules are applied to each cell in the grid simultaneously in each generation.

The idea is to "scan" the grid diagonally from the top-left to the bottom-left corner. The first two rules create a column of white cells next to a column of black cells, forming a wave-like domino effect that influences the bottom cells, making them valid for the first two rules in the next iteration. The purpose of the third rule is to maintain existing "good" columns and preserve the state of each cell until the "wave" reaches it.

The diagonal wave is crucial in a cyclic grid, enhancing the automaton's ability to converge more effectively compared to a top-bottom or left-right scan.

Although this deterministic automaton doesn't always achieve perfect black-and-white columns, as some runs may encounter conflicts in the cyclic grid, all runs strive to create zebra stripes. This diagonal approach improves convergence, even if not all attempts are flawless.

## The scale

First, we calculate the proportion of black cells in each column. If the proportion is greater than half, we calculate Ci=  (black cells)/(column size)  ​. If the proportion is less than or equal to half, we calculate Ci=1-  (black cells)/(column size)​.

Next, we introduce a penalty for adjacent columns that have similar patterns. Specifically, if two adjacent columns are both mostly black or both mostly white, we apply a penalty to our final scale.

Finally, we compute the average of the ​ values across all columns and adjust this average by the penalty factor to obtain the final scale.

### Explanation

**Proportion Calculation**: For each column, calculate the proportion of black cells. If it's greater than half, use the proportion directly. If it's less than or equal to half, use the complement of the proportion.

**Penalty Mechanism**: For each pair of adjacent columns, check if both are mostly black or both are mostly white. If they are, increment a penalty counter.

**Final Scale Calculation**: Compute the average of the calculated proportions (𝐶𝑖 values) and adjust this average by subtracting a normalized penalty factor. The penalty factor is the number of penalized adjacent pairs divided by the total number of adjacent pairs.

![image](https://github.com/user-attachments/assets/706c8378-9d6c-4e9f-a408-17aa67a56fba)

This scale is ranged from 0.5 (Chess board) to 1.0. we can see in the example above that over 10 runs of 250 iterations,  90% of the runs reached a perfect shape, but one of the runs converged to a “bad” shape that formed a half black half white columns, this is why it scale is only 6.175 (but still zebra shaped).

![image](https://github.com/user-attachments/assets/be0d72ed-1b9d-4629-830d-287232a5d244)


## Difference Between Runs

The variance graph provided shows how much the average proportion of black cells (between runs) deviates from the expected proportion (0.5 for a zebra pattern) over a course of 250 iterations.

* A higher a variance on the graph indicates that the cellular automaton produced stripes with a more uneven distribution of black and white cells across different runs.
  
* Conversely, a lower variance suggests more consistent pattern of black and white stripes across different runs.

![image](https://github.com/user-attachments/assets/72b088d2-9f39-4133-ad03-73c3210f008b)

The cellular automaton is deterministic, meaning it should always produce the same output for a given initial state. However, the graph shows some variation between the runs. This due to:

1. Initial configuration: because the initial configuration of black and white cells is randomized, it takes different number of iterations for different runs to reach a stable pattern..
   
2. Cyclic grid effects: The rules mentioned above are valid to a cyclic grid, which means the edges of the grid connect. The way this connection is handled influences how the wave propagates across the grid and leads to slight variations between runs.

   
Considering the two points, the variance graph shows that in the first 50 iterations, the variance decreases rapidly. This is likely due to the "wave strategy" that quickly corrects the initial configuration. The next 150 iterations exhibit a relatively small change in variance, indicating minor "noise" in the data but no significant change in the overall trend. However, in the last 50 iterations, the graph shows a very steep decline. This rapid drop can be attributed to the different waves likely merging into a single, unified wave. This unification process leads the variance to approach (almost) 0.

*Note: The variance doesn't reach absolute 0 because, as we mentioned, there is a single non-converging run causing a slight deviation.

## Dependencies

* Python 3.7+
* NumPy
* Matplotlib

## Installing And Executing
  
You can use [Git](https://git-scm.com). From your command line:

```bash
# Clone this repository.
$ git clone https://github.com/TalMizrahii/Cellular-Automata

# Go into the repository.
$ cd Cellular-Automata

# Run the program
$ cellularAutomata.exe

```
## Author

* [@Tal Mizrahi](https://github.com/TalMizrahii)
* Taltalon1927@gmail.com
