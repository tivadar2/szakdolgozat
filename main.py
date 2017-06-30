# coding=utf-8

import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from random import uniform, shuffle
from genetic import Population

# Program kezdete: if __name__ == '__main__':-nél lent

def get_nice_string(list_or_iterator):
    return "[" + ", ".join( str(x) for x in list_or_iterator) + "]"


def read_ages():
    age_file = open('gender_age.dat')
    lines = age_file.readlines()
    ages_list = []
    for line in lines:
        tmp = line.split(' ')
        id_age = [int(tmp[0]), int(tmp[2])]
        ages_list.append(id_age)
    age_file.close()
    ages = dict(ages_list)
    del ages_list
    del lines
    del age_file
    return ages


def read_groups(ego):
    file = open('60-69/ego' + str(ego) + '_out.dot')
    lines = file.readlines()
    groups = []
    for line in lines:
        tmp = line.split(',')
        nodes = []
        del tmp[-1]
        for node in tmp:
            nodes.append(int(node[1:]))
        groups.append(nodes)
    file.close()
    del file
    return groups


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class Group(object):
    def __init__(self, ages=(), group_size=-1, average_age=-1, deviation=-1, real_group_size=-1):
        self.ages = ages
        self.realGroupSize = real_group_size
        self.groupSize = group_size
        self.deviation = deviation
        self.averageAge = average_age

    def calculate_stats(self):
        average = 0
        wrong_ages_counter = 0
        for age in self.ages:
            if age == 999 or age == 555:
                wrong_ages_counter += 1
            else:
                average += age

        self.realGroupSize = len(self.ages)
        self.groupSize = self.realGroupSize - wrong_ages_counter
        if self.groupSize == 0:
            return      # Nem ismerjük senkinek sem a korát a csoportban
        self.averageAge = average/self.groupSize
        deviation = 0
        for age in self.ages:
            if not (age == 999 or age == 555):
                deviation += (self.averageAge - age)**2

        self.deviation = deviation/self.groupSize


class Ego(object):
    def __init__(self, id, real_age):
        self.ID = id
        self.realAge = real_age
        self.groups = []
        self.estimatedAge = -1


class MyApplication(object):
    def __init__(self):
        # Globális változók
        self.allAges = {}
        self.groups_of_ego = {}
        self.egos = []
        self.smoothedAgeDistribution = []
        self.class_egos = []
        self.q = 1  # Az összes ego hányadára számolja ki a becsléseket
                    # pl: self.q = 2 -> csak az egók felére végez becslést, ezáltal kb. kétszer gyorsabb

        # Paraméterek
        self.params = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1], dtype=np.float32)

    def make_histogram(self, ego):
        x = np.linspace(0, 80, 81)
        matrix = np.empty((len(ego.groups), 81))
        for num, g in enumerate(ego.groups):
            if g.averageAge < 0:
                continue
            sigma = g.deviation
            groupSize = g.groupSize

            suly = 1
            # Súlyok a különböző csoportméretek és csoportszórások esetén
            if sigma == 0 and groupSize <= 1:
                suly = self.params[9]
            elif sigma == 0 and groupSize > 1:
                suly = self.params[10]
            elif 0 < sigma <= 3 and groupSize <= 5:
                suly = self.params[0]
            elif 3 < sigma <= 6 and groupSize <= 5:
                suly = self.params[1]
            elif 6 < sigma and groupSize <= 5:
                suly = self.params[2]
            elif 0 < sigma <= 3 and 5 < groupSize <= 10:
                suly = self.params[3]
            elif 3 < sigma <= 6 and 5 < groupSize <= 10:
                suly = self.params[4]
            elif 6 < sigma and 5 < groupSize <= 10:
                suly = self.params[5]
            elif 0 < sigma <= 3 and 10 < groupSize:
                suly = self.params[6]
            elif 3 < sigma <= 6 and 10 < groupSize:
                suly = self.params[7]
            elif 6 < sigma and 10 < groupSize:
                suly = self.params[11]

            matrix[num] = suly*np.exp(-np.square(x - g.averageAge) / (2 * self.params[8]*self.params[8]))

        K = matrix.sum(axis=0)/self.smoothedAgeDistribution
        return K

    def get_fwhm(self, hist, age):
        half_max = hist[age] / 2
        left_half = 0
        right_half = 100
        for i in range(1, 100):  # Megkeresi a bal félérték helyét
            if age - i == 0:  # Túlment az index
                left_half = age - i
                break
            if hist[age - i] < half_max:
                left_half = age - i
                break
        for i in range(1, 100):  # Megkeresi a jobb félérték helyét
            if age + i == 80:  # Túlment az index
                left_half = age + i
                break
            if hist[age + i] < half_max:
                right_half = age + i
                break
        fwhm = right_half - left_half
        return fwhm

    def estimate_age(self, ego, debug=False):
        K = self.make_histogram(ego)

        peak_age = []
        peak_v = []
        for x in range(14, 80):
            if K[x] > K[x - 1] and K[x] > K[x + 1]:
                peak_age.append(x)
                h = K[x]
                w = self.get_fwhm(K, x)
                v = h / w
                peak_v.append(v)

        if len(peak_v) == 0:  # nincs csúcs
            # print(ego)
            return -1
        if debug:
            print(peak_age, peak_v)
        best_peak_index = peak_v.index(max(peak_v))
        estimated_age = peak_age[best_peak_index]
        # estimated_age = min(peak_age, key=lambda x: abs(x - ego.realAge))  # 50.7%-ig megy így
        return estimated_age

    def read_groups_from_dots(self):
        groups_of_ego = []
        for ego in self.egos:
            groups_of_ego.append([ego, read_groups(ego)])
        return dict(groups_of_ego)

    def make_smooth_age_distr(self):
        n = [0]*91
        index = list(range(10, 101))
        N = []
        values = list(self.allAges.values())
        for i in index:
            N.append(values.count(i))
        for i in range(91):
            for j in range(91):
                n[i] += gauss(i - j)*N[j]   # (i+10) - (j+10) = i-j
        return n

    def estimate_all_ages(self, parameters=None):
        pm1 = 0  # pm = +- (plus-minus)
        pm2 = 0
        pm3 = 0
        pm4 = 0
        pm5 = 0
        if parameters is not None:
            self.params = parameters
        for i in range(len(self.params)):  # ne legyen negatív súly
            if self.params[i] < 0:
                self.params[i] = 0
        shuffle(self.class_egos)
        for ego in self.class_egos[:len(self.class_egos)//self.q]:
            estimated_age = self.estimate_age(ego)
            ego.estimatedAge = estimated_age
            if estimated_age == -1:
                continue
            diff = pow((estimated_age - ego.realAge), 2)
            if diff <= 1:
                pm1 += 1
            if diff <= 4:
                pm2 += 1
            if diff <= 9:
                pm3 += 1
            if diff <= 16:
                pm4 += 1
            if diff <= 25:
                pm5 += 1

        return pm2/(len(self.class_egos)//self.q)

    def calc_derivative(self, x):
        grad = np.zeros(x.size)
        f_0 = self.estimate_all_ages()
        print("fitness: " + str(f_0))
        dx = 1
        for dim in range(x.size):
            x[dim] += dx
            grad[dim] = (self.estimate_all_ages() - f_0)/dx
            x[dim] -= dx
        return grad

    def simulated_annealing(self, cycles, start, no):
        if start == 'fix':
            self.params = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1], dtype=np.float32)
        elif start == 'random':
            self.params = np.array(
                [uniform(0, 5), uniform(0, 5), uniform(0, 5), uniform(0, 5), uniform(0, 5), uniform(0, 5),
                 uniform(0, 5), uniform(0, 5), uniform(0, 5), uniform(0, 5), uniform(0, 5), uniform(0, 5)])

        E = self.estimate_all_ages()

        prev_params = self.params
        prev_E = E

        T = 0.01
        with open("sa_"+str(cycles)+"_"+str(start)+"_"+str(no)+".log", "w+") as file:
            print("#T, fitness, params", file=file)
            while T > 0:
                d_params = np.array(
                    [uniform(-1, 1), uniform(-1, 1), uniform(-1, 1), uniform(-1, 1), uniform(-1, 1), uniform(-1, 1),
                     uniform(-1, 1), uniform(-1, 1), uniform(-1, 1), uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)])
                d_params = d_params / np.linalg.norm(self.params)
                self.params = self.params + d_params
                E = self.estimate_all_ages()
                print(self.params)
                print("E = " + str(E))
                # print(str(T) + " " + str(E) + " " + get_nice_string(self.params), file=file)
                if E > prev_E:
                    prev_params = self.params
                    prev_E = E
                    print("fel")
                else:
                    p = math.exp((E - prev_E) / T)
                    if uniform(0, 1) <= p:
                        prev_params = self.params
                        prev_E = E
                        print("le " + str(p))
                    else:
                        self.params = self.params - d_params
                        print("újra " + str(p))
                        # pass # nem csinál semmit, újra random
                print(str(T) + " " + str(prev_E) + " " + str(E) + " " + get_nice_string(self.params), file=file)
                print("T = " + str(T) + "\n")
                T -= 0.01/cycles

    def genetic_algorithm(self, population_size, generations, mutation):
        population = Population(self.estimate_all_ages, n=population_size,
                                mutation=mutation, params_size=self.params.size)
        for i in range(generations):
            population.evolve()
        with open("ga_"+str(population_size)+"_"+str(generations)+"_"+str(mutation)+".log", "w+") as file:
            print("#generation, best_fitness_of_generation, best_params", file=file)
            for i in range(generations):
                print(str(i+1) + " " + str(population.best_fitnesses[i])
                      + " " + get_nice_string(population.best_params[i]), file=file)

    def gradient_method(self):
        #self.params = np.array(
        #    [1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1])  # opt, mint optimal, de az algoritmus végén lesz (/lehet) optimális
        gamma = 1

        grad = self.calc_derivative(self.params)
        prev_params = self.params
        prev_grad = grad
        self.params = self.params + grad * gamma

        counter = 0
        while np.linalg.norm(grad) >= 0.00001 or counter > 1000:
            start = time.clock()
            grad = self.calc_derivative(self.params)
            # gamma = np.inner((params - prev_params), (grad - prev_grad)) / np.linalg.norm(grad - prev_grad)**2
            prev_params = self.params
            prev_grad = grad
            print('grad = ' + str(grad))
            print("params = " + str(self.params))
            print("gamma: " + str(gamma))
            self.params = self.params + grad * gamma
            counter += 1
            print("time:" + str(time.clock() - start) + "\n")

        print(counter)
        print(self.estimate_all_ages(self.params))
        print(self.params)


if __name__ == '__main__':
    app = MyApplication()
    app.smoothedAgeDistribution = load_obj("smoothAgeDistribution_iwiw_sigma2")  # életkoreloszlás, amelyikkel leosztunk
    # app.smoothedAgeDistribution = load_obj("smoothAgeDistribution_tel_sigma2")
    app.class_egos = load_obj('iwiw_50_class_egos')  # itt lehet állítani, hogy melyik hálózatot használjuk
    # app.class_egos = load_obj('class_egos')

    if len(sys.argv) == 1:
        start = time.time()
        print(app.estimate_all_ages())  # Kiírja a sikerességet
        print(app.params)
        end = time.time()
        print(end-start)
    else:
        if sys.argv[1] == 'grad':
            app.gradient_method()
        elif sys.argv[1] == 'sa':
            # python3 main.py sa #cycles #start=fix,random #no
            app.simulated_annealing(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
        elif sys.argv[1] == 'ga':
            # python3 main.py ga #population_size #generation #mutation #which_egos #q
            app.class_egos = load_obj(sys.argv[5])
            app.q = int(sys.argv[6])
            app.genetic_algorithm(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))