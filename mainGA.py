from exceptions import *
from game import *
import time
import numpy as np


try:
    n = int(input("Write a number greater than 3 to be the board dimension and press (Enter)\n"))
    if n < 4:
        raise LessThanFourError(n)

except ValueError as err:
    exit("The value must be an integer.")
except LessThanFourError as err:
    exit(str(err))


strg = input("Type a solution for the {}-queens problem and press (Enter)\n".format(n))

try:
    initialSolution = np.fromiter(map(int, strg.split(' ')), dtype=int)

    if initialSolution.size != n:
        raise IncorrectInputLengthError(n)

    if np.min(initialSolution) < 0 or np.max(initialSolution) > n:
        raise IncorrectInputError(n)

except IncorrectInputLengthError as err:
    exit(str(err))
except IncorrectInputError as err:
    exit(str(err))
except ValueError as err:
    exit("The characters in the solution must be integers between 0 and {}.".format(n))

# ----------------------------------------------------------------------------------------------------- #

# 1. Generate initial Population P(t) at random;
game = Game(initialSolution)
solutions = np.empty(shape=(0, game.n), dtype=int)

# 2. Evaluate the fitness of each individual in P(t);
populationWithFitness = game.get_fitness_of_each_individual()

# 3. while (not termination condition) do
beginning, lastFound = time.time(), time.time()
while time.time() - lastFound < 60. and time.time() - beginning < 300.:

    # 4. Select parents, Pa(t) from P(t) based on their fitness in P(t);
    parents = game.select_parents(populationWithFitness)

    # Restart population
    oldPopulation = game.population
    game.population = np.delete(game.population, np.s_[0:game.population.size], axis=0)  # np.s_[a:b] --> slice(a,b)

    # 5. Apply crossover to create offspring from parents: Pa(t) ->O(t)
    for couple in parents:
        child1, child2 = game.apply_crossover(couple[0], couple[1])
        # 6. Apply mutation to the offspring: O(t) ->O(t)
        child1, child2 = game.apply_mutation(child1), game.apply_mutation(child2)
        # 7. Select population P(t+1) from current offspring O(t) and parents P(t);
        for newIndividual in [couple[0], couple[1], child1, child2]:
            if not any(np.array_equal(newIndividual, individual) for individual in game.population):
                game.population = np.append(game.population, [newIndividual], axis=0)

    # Fill population with individuals not selected in previous population
    for newIndividual in oldPopulation:
        if not len(game.population) < game.populationSize:
            break
        elif not any(np.array_equal(newIndividual, individual) for individual in game.population):
            game.population = np.append(game.population, [newIndividual], axis=0)
        else:
            continue

    # 8. Evaluate the fitness of each individual in P(t);
    populationWithFitness = game.get_fitness_of_each_individual()
    for individual in populationWithFitness:
        if individual[1] == 0 and not any(np.array_equal(individual[0], solution) for solution in solutions):
            lastFound = time.time()
            solutions = np.append(solutions, [individual[0]], axis=0)
            game.print_solution(individual[0])

print("Number of solutions found in {} seconds: {}".format(lastFound - beginning, len(solutions)))

# For the case: n = 18 | 2 5 7 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# Without numpy: 253.03 seconds | Solutions: 1426
# With numpy: 185.99 seconds | Solutions: 1426

# For the case: n = 18 | 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
# With Genetic Algorithm: 162.01 seconds | 18
