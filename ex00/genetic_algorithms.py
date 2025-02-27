# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    genetic_algorithms.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: josfelip <josfelip@student.42sp.org.br>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/27 14:50:25 by josfelip          #+#    #+#              #
#    Updated: 2025/02/27 14:50:30 by josfelip         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

#!/usr/bin/env python3
import random
import subprocess
import numpy as np
import argparse
from typing import List, Tuple

class Individual:
    """
    Represents an individual solution with 4 frequency values between 45.0 and 50.0
    and the corresponding fitness metrics for each location.
    """
    def __init__(self, frequencies=None):
        # Initialize with random frequencies if none provided
        if frequencies is None:
            self.frequencies = [random.uniform(45.0, 50.0) for _ in range(4)]
        else:
            self.frequencies = frequencies
        
        # Initialize fitness metrics for the 5 locations
        self.location_scores = {
            "dorms": 0.0,
            "kitchen": 0.0,
            "showers": 0.0,
            "cluster": 0.0,
            "entrance": 0.0
        }
        self.fitness = 0.0
    
    def evaluate(self, cellular_binary_path="./cellular"):
        """
        Evaluate the individual by calling the provided binary to get reliability indices
        for each location. Then calculate the overall fitness.
        """
        # Prepare the frequencies as command line arguments
        freq_args = [f"{freq:.2f}" for freq in self.frequencies]
        
        # Call the binary and capture its output
        try:
            cmd = [cellular_binary_path] + freq_args
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout
            
            # Parse the output to extract reliability indices
            lines = output.strip().split('\n')
            for line in lines[1:]:  # Skip first line ("Cellular quality")
                if ':' in line:
                    location, score = line.split(':', 1)
                    location = location.strip()
                    score = float(score.strip())
                    self.location_scores[location] = score
            
            # Calculate overall fitness (this can be customized based on priorities)
            self.fitness = sum(self.location_scores.values())
            
            return self.fitness
        except subprocess.CalledProcessError as e:
            print(f"Error executing binary: {e}")
            return 0.0
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return 0.0
    
    def __str__(self):
        """String representation of the individual for display purposes."""
        freqs = " ".join([f"{freq:.2f}" for freq in self.frequencies])
        scores = " | ".join([f"{loc}: {score:.3f}" for loc, score in self.location_scores.items()])
        return f"Frequencies: [{freqs}] | {scores} | Overall: {self.fitness:.3f}"


class GeneticAlgorithm:
    """
    Implements a genetic algorithm to optimize antenna frequency settings.
    """
    def __init__(self, 
                 population_size=1000, 
                 mutation_rate=0.1, 
                 crossover_rate=0.8,
                 elite_size=20,
                 tournament_size=5,
                 cellular_binary="./cellular"):
        """
        Initialize the genetic algorithm with the given parameters.
        
        Args:
            population_size: Number of individuals in the population
            mutation_rate: Probability of mutation for each gene (frequency)
            crossover_rate: Probability of crossover between two parents
            elite_size: Number of top individuals to preserve unchanged
            tournament_size: Number of individuals in tournament selection
            cellular_binary: Path to the cellular quality evaluation binary
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.cellular_binary = cellular_binary
        self.population = []
        self.generation = 0
        self.best_individual = None
    
    def initialize_population(self):
        """Create the initial population of random individuals."""
        self.population = [Individual() for _ in range(self.population_size)]
        self.generation = 0
    
    def evaluate_population(self):
        """Evaluate all individuals in the population."""
        for individual in self.population:
            individual.evaluate(self.cellular_binary)
        
        # Sort population by fitness in descending order
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        
        # Update best individual if needed
        if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = Individual(self.population[0].frequencies)
            self.best_individual.location_scores = self.population[0].location_scores.copy()
            self.best_individual.fitness = self.population[0].fitness
    
    def tournament_selection(self) -> Individual:
        """
        Select an individual using tournament selection.
        
        Returns:
            The selected individual
        """
        # Randomly select tournament_size individuals
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        # Return the best individual from the tournament
        return max(tournament, key=lambda ind: ind.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Perform crossover between two parents to create two offspring.
        
        Args:
            parent1, parent2: The parent individuals
            
        Returns:
            A tuple of two offspring individuals
        """
        if random.random() > self.crossover_rate:
            # No crossover, just clone the parents
            return (
                Individual(parent1.frequencies.copy()),
                Individual(parent2.frequencies.copy())
            )
        
        # Perform uniform crossover
        child1_freqs = []
        child2_freqs = []
        
        for i in range(len(parent1.frequencies)):
            if random.random() < 0.5:
                child1_freqs.append(parent1.frequencies[i])
                child2_freqs.append(parent2.frequencies[i])
            else:
                child1_freqs.append(parent2.frequencies[i])
                child2_freqs.append(parent1.frequencies[i])
        
        return Individual(child1_freqs), Individual(child2_freqs)
    
    def mutate(self, individual: Individual):
        """
        Apply mutation to an individual.
        
        Args:
            individual: The individual to mutate
        """
        for i in range(len(individual.frequencies)):
            if random.random() < self.mutation_rate:
                # Apply gaussian mutation (with 10% of the range as standard deviation)
                mutation_sigma = 0.5  # 10% of the range 45.0-50.0
                new_value = individual.frequencies[i] + random.gauss(0, mutation_sigma)
                # Ensure the new value stays within the allowed range
                individual.frequencies[i] = max(45.0, min(50.0, new_value))
    
    def create_next_generation(self):
        """Create the next generation through selection, crossover, and mutation."""
        new_population = []
        
        # Elitism: Keep the best individuals unchanged
        elites = self.population[:self.elite_size]
        new_population.extend([Individual(elite.frequencies) for elite in elites])
        
        # Create the rest of the population through selection, crossover, and mutation
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation
            self.mutate(child1)
            self.mutate(child2)
            
            # Add to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = new_population
        self.generation += 1
    
    def run(self, max_generations=100, target_fitness=None):
        """
        Run the genetic algorithm for a specified number of generations or until
        a target fitness is reached.
        
        Args:
            max_generations: Maximum number of generations to run
            target_fitness: Optional target fitness to stop early
        """
        self.initialize_population()
        
        for generation in range(max_generations):
            self.evaluate_population()
            
            best_individual = self.population[0]
            avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
            
            # Display progress
            print(f"Generation {self.generation}:")
            print(f"  Best: {best_individual}")
            print(f"  Avg Fitness: {avg_fitness:.3f}")
            
            # Check if target fitness reached
            if target_fitness is not None and best_individual.fitness >= target_fitness:
                print(f"Target fitness {target_fitness} reached!")
                break
            
            # Create next generation (unless this is the last iteration)
            if generation < max_generations - 1:
                self.create_next_generation()
        
        print("\nBest solution found:")
        print(self.best_individual)
        
        # Return the best frequencies found
        return self.best_individual.frequencies


def main():
    """Main function to parse arguments and run the genetic algorithm."""
    parser = argparse.ArgumentParser(description='Optimize antenna frequencies using a genetic algorithm')
    parser.add_argument('--population', type=int, default=1000, help='Population size')
    parser.add_argument('--generations', type=int, default=100, help='Maximum number of generations')
    parser.add_argument('--mutation-rate', type=float, default=0.1, help='Mutation rate')
    parser.add_argument('--crossover-rate', type=float, default=0.8, help='Crossover rate')
    parser.add_argument('--elite-size', type=int, default=20, help='Number of elite individuals')
    parser.add_argument('--tournament-size', type=int, default=5, help='Tournament size for selection')
    parser.add_argument('--target-fitness', type=float, help='Target fitness to stop early')
    parser.add_argument('--binary', default='./cellular', help='Path to the cellular binary')
    
    args = parser.parse_args()
    
    # Create and run the genetic algorithm
    ga = GeneticAlgorithm(
        population_size=args.population,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        elite_size=args.elite_size,
        tournament_size=args.tournament_size,
        cellular_binary=args.binary
    )
    
    best_frequencies = ga.run(
        max_generations=args.generations,
        target_fitness=args.target_fitness
    )
    
    # Print final result
    print("\nOptimal Frequency Settings:")
    print(" ".join([f"{freq:.2f}" for freq in best_frequencies]))


if __name__ == "__main__":
    main()