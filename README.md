# Genetic Algorithm for Antenna Frequency Optimization

## Overview
This project implements a genetic algorithm to optimize antenna frequency settings for improving cellular reception across multiple locations within Forty2, a historic fort on the west coast of France. The algorithm simulates natural evolution to find the optimal frequency settings that provide the best overall cellular coverage balance.

## Problem Description
Forty2 is a place where students and alumni gather to code and relax, but it suffers from suboptimal cellular reception in various locations. This project aims to optimize the frequency settings of four antennas to improve cellular reliability across five key locations:
- Dormitories
- Kitchen
- Showers
- Cluster
- Entrance

The antenna frequencies must be within the range of 45.0 to 50.0 as specified in the requirements.

## How It Works
The genetic algorithm works by:
1. Generating an initial population of random frequency settings
2. Evaluating each individual's performance using an external binary
3. Selecting the best-performing settings through tournament selection
4. Creating new settings through crossover and mutation
5. Repeating this process over multiple generations to converge on optimal settings

## Features
- Customizable population size, mutation rate, crossover rate, and other genetic algorithm parameters
- Elitism to preserve the best solutions between generations
- Tournament selection for parent selection
- Uniform crossover and Gaussian mutation
- Early stopping when a target fitness is reached
- Progress tracking and reporting

## Requirements
- Python 3.6 or higher
- NumPy
- The `cellular` binary (for evaluating frequency settings)

## Installation
1. Clone the repository
2. Ensure Python and NumPy are installed
3. Place the `cellular` binary in the same directory as the script or specify its path

## Usage
Run the script with default parameters:
```bash
python genetic_algorithms.py
```

Customize the genetic algorithm parameters:
```bash
python genetic_algorithms.py --population 2000 --generations 200 --mutation-rate 0.15 --crossover-rate 0.9 --elite-size 30 --tournament-size 7
```

Available parameters:
- `--population`: Size of the population (default: 1000)
- `--generations`: Maximum number of generations to run (default: 100)
- `--mutation-rate`: Probability of mutation for each gene (default: 0.1)
- `--crossover-rate`: Probability of crossover between two parents (default: 0.8)
- `--elite-size`: Number of top individuals to preserve (default: 20)
- `--tournament-size`: Number of individuals in tournament selection (default: 5)
- `--target-fitness`: Target fitness to stop early (optional)
- `--binary`: Path to the cellular binary (default: ./cellular)

## Implementation Details

### Individual Class
Represents a potential solution with four frequency values (45.0-50.0) and their evaluated performance metrics.

Key methods:
- `__init__`: Initialize with random or specified frequencies
- `evaluate`: Call the external binary to get reliability indices for each location
- `__str__`: String representation for display

### GeneticAlgorithm Class
Implements the genetic algorithm logic.

Key methods:
- `initialize_population`: Create the initial population of random individuals
- `evaluate_population`: Evaluate all individuals and update the best solution
- `tournament_selection`: Select an individual using tournament selection
- `crossover`: Create two offspring through uniform crossover of parents
- `mutate`: Apply Gaussian mutation to an individual
- `create_next_generation`: Create the next generation through selection, crossover, and mutation
- `run`: Run the genetic algorithm for a specified number of generations

## Fitness Calculation
The overall fitness of an individual is calculated as the sum of reliability indices across all five locations. To encourage balanced solutions, individuals with a lower standard deviation (better balance) receive a bonus to their fitness score.

## Example Output
```
Generation 0:
  Best: Frequencies: [45.35 49.72 47.31 46.19] | dorms: 3.123 | kitchen: 1.932 | showers: 2.741 | cluster: 3.289 | entrance: 2.195 | Overall: 13.280
  Avg Fitness: 9.873

Generation 1:
  Best: Frequencies: [45.39 49.65 47.28 46.11] | dorms: 3.245 | kitchen: 2.108 | showers: 2.793 | cluster: 3.312 | entrance: 2.305 | Overall: 13.763
  Avg Fitness: 10.456

...

Generation 99:
  Best: Frequencies: [45.41 49.61 47.23 46.08] | dorms: 3.542 | kitchen: 2.871 | showers: 3.145 | cluster: 3.498 | entrance: 2.986 | Overall: 16.042
  Avg Fitness: 14.893

Best solution found:
Frequencies: [45.41 49.61 47.23 46.08] | dorms: 3.542 | kitchen: 2.871 | showers: 3.145 | cluster: 3.498 | entrance: 2.986 | Overall: 16.042

Optimal Frequency Settings:
45.41 49.61 47.23 46.08
```

## Notes
- The fitness evaluation relies on an external binary (`cellular`) to simulate reception quality
- Frequency values are constrained between 45.0 and 50.0
- The algorithm aims to balance good reception across all locations rather than maximizing a single location

## Author
- josfelip@student.42sp.org.br