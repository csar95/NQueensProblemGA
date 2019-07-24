import numpy as np
import random


class Game:

    # Instance variable
    populationSize = 0

    # Class variable
    accumulatedProbability = 0.

    def __init__(self, solution):
        self.n = solution.size
        self.population = solution.reshape((1, self.n))
        self.populationSize = self.n*2
        self.generate_initial_population()

    def generate_initial_population(self):
        while self.population.shape[0] < self.populationSize:
            newChromosome = np.arange(1, self.n + 1)
            np.random.shuffle(newChromosome)
            if not any(np.array_equal(newChromosome, chromosome) for chromosome in self.population):
                self.population = np.append(self.population, newChromosome.reshape((1, self.n)), axis=0)

    def get_fitness_of_each_individual(self):
        return np.array([(chromosome, self.evaluate_fitness(chromosome)) for chromosome in self.population],
                        dtype=[('chromosome', object), ('fitness', int)])

    # Explanation in https://arxiv.org/pdf/1802.02006.pdf
    def evaluate_fitness(self, chromosome):
        t1 = 0  # Number of repetitive queens in one diagonal while seen from left corner
        t2 = 0  # Number of repetitive queens in one diagonal while seen from right corner

        f1 = np.array([chromosome[i] - (i + 1) for i in range(self.n)])
        f2 = np.array([(1 + self.n) - chromosome[i] - (i + 1) for i in range(self.n)])

        f1 = np.sort(f1)
        f2 = np.sort(f2)

        for i in range(1, self.n):
            if f1[i] == f1[i - 1]:
                t1 += 1
            if f2[i] == f2[i - 1]:
                t2 += 1

        return t1 + t2

    def select_parents(self, populationWithFitness):
        # Select maximum fitness value among all individual in population
        maxFitness = np.amax(populationWithFitness['fitness'])
        for individual in populationWithFitness:
            individual[1] = maxFitness - individual[1]
        totalFitness = np.sum(populationWithFitness['fitness'])

        # Find probability of being chosen for each individual in population based on its fitness value
        populationWithSelectionProb = np.array(
            [(chromosome[0], Game.get_probability(chromosome[1], totalFitness)) for chromosome in populationWithFitness],
            dtype=[('chromosome', object), ('probability', float)])
        populationWithSelectionProb[-1][1] = 1.
        Game.accumulatedProbability = 0.

        # Apply Roulette Wheel Selection method for chosing parents in the population
        parents = []
        while len(parents) < self.populationSize/4:
            p1 = random.random()
            p2 = p1 + .5 if p1 + .5 < 1. else p1 + .5 - 1.
            chromosome1, chromosome2 = np.zeros(self.n, dtype=int), np.zeros(self.n, dtype=int)
            for individual in populationWithSelectionProb:
                if p1 < individual[1] and not all(chromosome1):
                    chromosome1 = individual[0]
                    if all(chromosome2): break
                elif p2 < individual[1] and not all(chromosome2):
                    chromosome2 = individual[0]
                    if all(chromosome1): break
            parents.append((chromosome1, chromosome2))

        return np.array(parents, dtype=[('chromosome1', object), ('chromosome2', object)])

    @classmethod
    def get_probability(cls, fitness, totalFitness):
        cls.accumulatedProbability += fitness / totalFitness
        return cls.accumulatedProbability

    # Order One Crossover
    def apply_crossover(self, parent1, parent2):
        child1 = np.zeros(self.n, dtype=int)
        child2 = np.zeros(self.n, dtype=int)

        cutPoint1 = random.randint(0, self.n - 2)
        cutPoint2 = random.randint(cutPoint1 + 1, self.n - 1)

        for i in range(cutPoint1, cutPoint2 + 1):
            child1[i] = parent1[i]
            child2[i] = parent2[i]

        child1Index = cutPoint2
        child2Index = cutPoint2

        for i in range(cutPoint2, cutPoint2 + self.n):
            i = i % self.n
            if parent2[i] not in child1:
                child1Index += 1
                child1[child1Index % self.n] = parent2[i]
            if parent1[i] not in child2:
                child2Index += 1
                child2[child2Index % self.n] = parent1[i]

        return child1, child2

    def apply_mutation(self, chromosome):
        if random.uniform(0, 1) > 0.8:
            positions = list(range(self.n))
            random.shuffle(positions)
            pos1 = positions.pop()
            pos2 = positions.pop()
            chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]
        return chromosome

    def print_solution(self, solution):
        result = ""
        for col in range(self.n):
            result += (str(solution[col]) + ' ')
        print(result)
