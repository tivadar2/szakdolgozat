import numpy as np
import random
from random import choice, shuffle


class Individual(object):
    def __init__(self, func, number_of_args=0, params=None, fitness=0):
        self.fitnessFunction = func
        if params is None:  # random paraméterek
            self.params = np.random.uniform(0, 5, number_of_args)
        else:
            self.params = params
            self.fitness = fitness

    def calc_fitness(self):
        self.fitness = self.fitnessFunction(self.params)


class Population(object):
    def __init__(self, func, params_size, n=0, individuals=None, mutation=0.1):
        self.fitnessFunction = func
        self.mutation = mutation
        self.best_fitnesses = []
        self.best_params = []
        if individuals is None:  # Random generálunk egyedeket, ha nincsenek megadva
            self.individuals = [Individual(self.fitnessFunction, number_of_args=params_size) for i in range(n)]
            self.n = n
        else:
            self.individuals = individuals
            self.n = len(individuals)
        self.generation = 1

    def calc_fitness(self):  # Minden egyedre kiszámítjuk a fitnesst, esetünkben a becslés sikerességét
        for indiv in self.individuals:
            indiv.calc_fitness()

    def cross(self, parent1, parent2):  # Keresztezés: random összefésüli a szülők paramétereit
        number_of_params = len(parent1.params)
        params = np.zeros(number_of_params)
        for i in range(number_of_params):
            params[i] = choice([parent1.params[i], parent2.params[i]])
        child = Individual(self.fitnessFunction, params=params)
        self.individuals.append(child)

    def crossover(self):  # Az összes egyedet keresztezi random másikkal
        all_individual = list(range(len(self.individuals)))
        shuffle(all_individual)
        while len(all_individual) != 0:
            parent_a = self.individuals[all_individual.pop()]
            parent_b = self.individuals[all_individual.pop()]
            self.cross(parent_a, parent_b)

    def mutate(self):  # Mutáció: az összes egyed összes paraméterének bizonyos százalékát módosítja
        for indiv in self.individuals:
            for i in range(len(indiv.params)):
                if random.random() < self.mutation:
                    indiv.params[i] += np.random.normal(scale=0.5)

    def selection(self):  # Szelekció, kiválasztódás
        self.individuals.sort(key=lambda individual: individual.fitness, reverse=True)
        del self.individuals[self.n:]

    def evolve(self):  # "Evolúció"
        self.crossover()
        self.mutate()
        self.calc_fitness()
        self.selection()

        self.best_fitnesses.append(self.individuals[0].fitness)
        self.best_params.append(self.individuals[0].params)
        print("Generation " + str(self.generation))
        for indiv in self.individuals:
            print("fitness: " + str(indiv.fitness) + "  params: " + str(indiv.params))
        print('\n')
        self.generation += 1
