import numpy as np
from numpy.random import rand, randint
from tqdm import tqdm
import matplotlib.pyplot as plt


def objective_function(x: int) -> float:
    return (x ** (1 / 2)) * np.sin(10 * x)


def crossover(population: list, crossover_rate: float) -> list:
    offspring = list()
    for i in range(len(population) // 2):
        p1 = population[2 * i - 1].copy()  # parent 1
        p2 = population[2 * i].copy()  # parent 2
        if rand() < crossover_rate:
            cp = randint(0, len(p1) - 1, size=2)  # two random cutting points
            while cp[0] == cp[1]:
                cp = randint(0, len(p1) - 1, size=2)

            cp = sorted(cp)
            c1 = p1[:cp[0]] + p2[cp[0]:cp[1]] + p1[cp[1]:]
            c2 = p2[:cp[0]] + p1[cp[0]:cp[1]] + p2[cp[1]:]
            offspring.append(c1)
            offspring.append(c2)
        else:
            offspring.append(p1)
            offspring.append(p2)

    return offspring


def mutation(population: list, mutation_rate: float) -> list:
    offspring = list()
    for i in range(len(population)):
        parent = population[i].copy()
        if rand() < mutation_rate:
            cp = randint(0, len(parent))  # random cutting point
            c1 = parent
            if c1[cp] == 1:
                c1[cp] = 0  # flip
            offspring.append(c1)
        else:
            offspring.append(parent)

    return offspring


def selection(population: list, fitness: list, population_size: int) -> list:
    next_generation = list()
    elite = np.argmax(fitness)
    next_generation.append(population[elite])  # keep the best
    abs_fitness = [abs(fitness[i]) for i in range(len(fitness))]
    probability = [f / sum(abs_fitness) for f in abs_fitness]  # selection probability
    index = list(range(len(population)))
    index_selection = np.random.choice(
        index,
        size=len(population),
        replace=True,
        p=probability)
    for i in range(0, population_size - 1):
        next_generation.append(population[index_selection[i]])

    return next_generation


def decoding(bounds: list, bits: int, chromosome: list) -> float:
    st, en = 0, bits  # extract the chromosome
    sub = chromosome[st:en]
    chars = ''.join([str(s) for s in sub])
    integer = int(chars, 2)
    real_value = bounds[0] + (integer / bits ** 2) * (bounds[1] - bounds[0])
    return real_value


def main() -> None:
    # Parameters of the binary genetic algorithm
    bounds = [0, 6]
    iteration = 50
    bits = 8
    population_size = 50
    crossover_rate = 0.7
    mutation_rate = 0.3

    # Initial population
    population = [randint(
        low=0,
        high=2,
        size=bits
    ).tolist() for _ in range(population_size)]

    best_fitness = list()
    for _ in tqdm(range(0, iteration), desc="Graph construction: "):
        offspring = crossover(
            population=population,
            crossover_rate=crossover_rate,
        )
        offspring = mutation(
            population=offspring,
            mutation_rate=mutation_rate,
        )

        for s in offspring:
            population.append(s)

        real_chromosome = [decoding(bounds, bits, p) for p in population]
        fitness = [objective_function(x) for x in real_chromosome]
        best_fitness.append(max(fitness))
        best_fitness.append(min(fitness))
        population = selection(population, fitness, population_size)

    print(f'Min objective function value: {min(best_fitness)}')
    print(f'Max objective function value: {max(best_fitness)}')
    fig = plt.figure()
    plt.plot(best_fitness)
    fig.suptitle('Evolution of the best chromosome')
    plt.xlabel('Iteration * 2')
    plt.ylabel('Objective function value')
    plt.show()


if __name__ == '__main__':
    main()
