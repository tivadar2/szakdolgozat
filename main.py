# coding=utf-8

import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from random import uniform, shuffle
from genetic import Population
import cProfile
from os import listdir
import gzip
import re


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

"""
files = listdir('60-69')
for filename in files:
    if '_out.dot' in filename:
        ego = int(''.join(list(filter(str.isdigit, filename))))
        egos.append(ego)
"""
# Saját exp
own_exp = []
sigma = 2
own_exp.append([0, math.exp(-math.pow(0, 2) / (2 * math.pow(sigma, 2)))])
for i in range(1, 100):
    own_exp.append([i, math.exp(-math.pow(i, 2) / (2 * math.pow(sigma, 2)))])
    own_exp.append([-i, math.exp(-math.pow(i, 2) / (2 * math.pow(sigma, 2)))])
own_exp = dict(own_exp)


def gauss(x):
    return own_exp[int(x+0.5)]
    # return math.exp(-x*x/(2*sigma**2))


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
        self.q = 1

        # Paraméterek
        self.params = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1], dtype=np.float32)
        #self.params = np.array([6.7747038649, 7.71053429695, 0.379544637336, 3.19092356402, 3.96399904303, -4.47303253856, 3.49076237822,
         #1.71134368002, 2.67, 4.32734474146, 8.22419206133])

    def make_histogram(self, ego):
        x = np.linspace(0, 80, 81)
        matrix = np.empty((len(ego.groups), 81))
        for num, g in enumerate(ego.groups):
            if g.averageAge < 0:
                continue
            # if sigma == 0:
            # devs[0] += 1  # 589 ezer ilyen van!!!
            sigma = g.deviation
            groupSize = g.groupSize

            suly = 1
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

        # plt.plot(x, K)
        # plt.show()
        return K

    def get_fwhm(self, hist, age):  # TODO: Ezen még lehetne javítani
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
                # if x-1 in range(1, 80) and x-2 in range(1, 80) and x+1 in range(1, 80) and x+2 in range(1, 80):
                #    v = (K[x]-K[x-1]) + (K[x]-K[x+1]) + (K[x]-K[x+2]) + (K[x]-K[x-2])
                peak_v.append(v)

        if len(peak_v) == 0:
            # print(ego) TODO:
            return -1
        if debug:
            print(peak_age, peak_v)
        best_peak_index = peak_v.index(max(peak_v))  # TODO: ha nem talál csúcsot, akkor mit tegyen? 104840 - üres fájl
        estimated_age = peak_age[best_peak_index]
        # estimated_age = min(peak_age, key=lambda x: abs(x - ego.realAge))  # 50.7%-ig megy így
        # TODO: csúcskiválasztás javítása
        # if math.fabs(estimated_age - ego.realAge) <= 2:
        #    if ego.ID == 82152624:
        #        with open("histogram_egycsucs.txt", "w+") as file:
        #            for i in range(81):
        #                print(i, K[i], file=file)
        #        plt.plot(np.linspace(0, 80, 81), K)
        #        plt.show()
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
        pm1 = 0
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

        #print("pm1", pm1/(len(self.class_egos)//self.q))
        #print("pm2", pm2 / (len(self.class_egos) // self.q))
        #print("pm3", pm3 / (len(self.class_egos) // self.q))
        #print("pm4", pm4 / (len(self.class_egos) // self.q))
        #print("pm5", pm5 / (len(self.class_egos) // self.q))
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
            self.params = np.array([1, 1, 1, 1, 1, 1,
                                    1, 1, 3, 1, 1])
        elif start == 'random':
            self.params = np.array(
                [uniform(0, 5), uniform(0, 5), uniform(0, 5), uniform(0, 5), uniform(0, 5), uniform(0, 5),
                 uniform(0, 5), uniform(0, 5), uniform(0, 5), uniform(0, 5), uniform(0, 5)])

        E = self.estimate_all_ages()

        prev_params = self.params
        prev_E = E

        T = 0.01
        with open("sa_"+str(cycles)+"_"+str(start)+"_"+str(no)+".log", "w+") as file:
            print("#T, fitness, params", file=file)
            while T > 0:
                d_params = np.array(
                    [uniform(-1, 1), uniform(-1, 1), uniform(-1, 1), uniform(-1, 1), uniform(-1, 1), uniform(-1, 1),
                     uniform(-1, 1), uniform(-1, 1), uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)])
                d_params = d_params / np.linalg.norm(self.params)
                self.params = self.params + d_params
                E = self.estimate_all_ages()
                print(self.params)
                print("E = " + str(E))
                print(str(T) + " " + str(E) + " " + get_nice_string(self.params), file=file)
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
                print("T = " + str(T) + "\n")
                T -= 0.01/cycles

    def genetic_algorithm(self, population_size, generations, mutation):
        population = Population(self.estimate_all_ages, n=population_size, mutation=mutation)
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

    """
    # iWiW adatok konvertálása az Ego class formátumba + pickle
    # files = listdir('dot')
    egos = []
    with open('ids50.dat') as ids_file:
        egos = [int(line) for line in ids_file.readlines()]
    # for filename in files:
    #    if '_out.dot' in filename:
    #        ego = int(''.join(list(filter(str.isdigit, filename))))
    #        egos.append(ego)

    allAges = {}
    with open('birthdate.dat') as birthdate_file:
        lines = birthdate_file.readlines()
    for line in lines:
        id_birthdate = [int(x) for x in line.split()]
        if id_birthdate[1] == 0 or id_birthdate[1] == 1900:
            continue
        allAges[id_birthdate[0]] = 2013 - id_birthdate[1]

    allAges = load_obj("allAges")
    ages = np.zeros(81)
    for age in range(81):
        ages[age] = list(allAges.values()).count(age)
    aaa = np.zeros(81)
    x = np.linspace(0, 80, 81)
    for age in range(81):
        aaa += (1/(math.sqrt(2*math.pi)*2))*np.exp(-np.square(x - age)/(2*4))*ages[age]
    ages = aaa/90000

    save_obj(ages, "smoothAgeDistribution_tel_sigma2")
    plt.plot(np.linspace(0, 80, 81), ages)
    plt.show()

    class_egos = []
    count = 0
    for ego in egos:
        try:
            if not 14 <= allAges[ego] < 80:
                continue
            class_ego = Ego(ego, allAges[ego])
        except KeyError:
            count += 1
            continue
        groups = []
        try:
            with gzip.open('dot/{}_out.dot.gz'.format(ego)) as file:
                lines = file.readlines()
                for line in lines:
                    ages = []
                    for j_id in str(line).split(',')[:-1]:
                        ego_in_group = int(re.findall('\d+', j_id)[0])
                        if ego_in_group != ego:
                            try:
                                ages.append(allAges[ego_in_group])
                            except KeyError:
                                ages.append(999)
                    group = Group(ages)
                    group.calculate_stats()
                    groups.append(group)
        except FileNotFoundError:
            continue
        class_ego.groups = groups
        class_egos.append(class_ego)
    print(count)
    print(len(class_egos))
    save_obj(class_egos, 'iwiw_50_class_egos')
    """

    app = MyApplication()
    app.smoothedAgeDistribution = load_obj("smoothAgeDistribution_iwiw_sigma2")
    app.class_egos = load_obj('iwiw_50_class_egos')
    # app.class_egos = load_obj('class_egos')
    """
    group_sizes = [0]*301
    for class_ego in app.class_egos:
        for group in class_ego.groups:
            group_sizes[group.groupSize] += 1
    plt.plot(np.linspace(0, 20, 21), group_sizes[:21])
    plt.show()

    group_devs = [0] * 21
    for class_ego in app.class_egos:
        for group in class_ego.groups:
            if group.deviation == 0:
                group_devs[0] += 1
                continue
            for i in range(1, 21):
                if (i-1) < group.deviation <= i:
                    group_devs[i] += 1
                    continue
    plt.plot(np.linspace(0, 20, 21), group_devs[:21])
    plt.show()
    """
    #app.params = np.array([1.22273272, 0.87814622, -0.57510987, 1.20235717, 1.04234918, 0.18717539,
    #                       1.05593288, 1.01957651, 1.88473831, 0.38054335, 1.22493008])

    if len(sys.argv) == 1:
        start = time.time()
        # app.params = [6.7747038649, 7.71053429695, 0.379544637336, 3.19092356402, 3.96399904303, -4.47303253856, 3.49076237822, 1.71134368002, 14.2718478954, 4.32734474146, 8.22419206133]
        print(app.estimate_all_ages())
        #with open("param0.txt", "w+") as outfile:
        #    for i in range(1, 100):
        #        app.params[0] = i/10
        #        print(i/10, app.estimate_all_ages())
        #        print(i/10, app.estimate_all_ages(), file=outfile)
        print(app.params)
        end = time.time()
        print(end-start)
        """
        ego_set = set()
        with open("tamas_pred/predictions_200.dat") as file:
            lines = file.readlines()
            c = 0
            for line in lines:
                age_tamas = int(line.split("\t")[0])
                estAge_tamas = int(line.split("\t")[1])
                id_tamas = int(line.split("\t")[2])
                for e in app.class_egos:
                    if e.ID == id_tamas:
                        ego_set.add(e)
                    if (e.ID == id_tamas and
                            estAge_tamas != e.estimatedAge and
                            estAge_tamas-1 != e.estimatedAge and
                            estAge_tamas+1 != e.estimatedAge):
                        pass
                        #print(e.ID, age_tamas, e.realAge, estAge_tamas, e.estimatedAge)
                        #app.estimate_age(e, True)
                        #plt.plot(np.linspace(0, 80, 81), app.make_histogram(e))
                        #plt.show()
                        c += 1
        print(c)
        print(len(app.class_egos))
        difference = set(app.class_egos) - ego_set
        print(len(difference), len(set(app.class_egos)), len(ego_set))
        howManySuccesful = 0
        allOfThem = 0
        for ego in difference:
            if math.fabs(ego.realAge - ego.estimatedAge) <= 2:
                howManySuccesful += 1
            allOfThem += 1
            print(ego.ID, ego.realAge, ego.estimatedAge)
        print(allOfThem, howManySuccesful)
        """
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

    # start = time.time()
    # print(estimate_all_ages())
    # end = time.time()
    # print(end-start)

    """
    percents = []
    for i in np.linspace(1, 10, 100):
        opt_sigma = i
        percents.append(estimate_all_ages())

    plt.plot(np.linspace(1, 10, 100), percents)
    plt.show()
    """