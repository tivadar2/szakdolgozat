# coding=utf-8

import math
import pickle
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
import sys
from random import shuffle, uniform
import time
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import pycuda.cumath
import skcuda.linalg
import skcuda.misc
skcuda.linalg.init()

# Globális változók
allAges = {}
groups_of_ego = {}
egos = []
smoothedAgeDistr = []
class_egos = []

# Paraméterek
opts = np.array([1, 1, 1, 1, 1, 1, 1, 1, 4])


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


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
    # del tmp            # TODO: valami hiba
    file.close()
    del file
    return groups

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

        self.realGroupSize = len(ages)
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


def make_histogram(ego):
    K = np.zeros(81).astype(np.float32)
    gpu_K = gpuarray.to_gpu(K)
    x = np.linspace(0, 80, 81).astype(np.float32)
    matrix = np.zeros((len(ego.groups), 81)).astype(np.float32)
    gpu_ones = gpuarray.to_gpu(np.ones(len(ego.groups)).astype(np.float32))
    for num, g in enumerate(ego.groups):
        # if sigma == 0:
        # devs[0] += 1  # 589 ezer ilyen van!!!
        # sigma = g.deviation
        """
        if sigma <= 3 and groupSize <= 5:
            suly = opts[0]
        elif 3 < sigma <= 6 and groupSize <= 5:
            suly = opts[1]
        elif 6 < sigma and groupSize <= 5:
            suly = opts[2]
        elif sigma <= 3 and 5 < groupSize <= 10:
            suly = opts[3]
        elif 3 < sigma <= 6 and 5 < groupSize <= 10:
            suly = opts[4]
        elif 6 < sigma and 5 < groupSize <= 10:
            suly = opts[5]
        elif sigma <= 3 and 10 < groupSize:
            suly = opts[6]
        elif 3 < sigma <= 6 and 10 < groupSize:
            suly = opts[7]
        elif 6 < sigma and 10 < groupSize:
            suly = opts[8]
        """

        suly = 1
        if sigma == 0:
            suly = opts[0]
        elif 0 < sigma <= 2.5:
            suly = opts[1]
        elif 2.5 < sigma <= 5:
            suly = opts[3]
        elif 5 < sigma <= 10:
            suly = opts[4]
        elif 10 < sigma <= 15:
            suly = opts[5]
        elif 15 < sigma <= 25:
            suly = opts[6]
        elif 25 < sigma:
            suly = opts[7]

        matrix[num] = x - g.averageAge
        # matrix[num] = suly*pycuda.cumath.exp(
        # -skcuda.misc.multiply(x - g.averageAge, x - g.averageAge) / (2 * opts[8]))

    gpu_mat = gpuarray.to_gpu(matrix)
    szorzo = np.float32((1/(2*opts[8])))
    gpu_mat = pycuda.cumath.exp(-skcuda.misc.multiply(gpu_mat, gpu_mat)*szorzo)
    # K = gpuarray.sum(gpu_mat).get()
    K = skcuda.linalg.dot(skcuda.linalg.transpose(gpu_mat), gpu_ones).get()
    # wait = input("PRESS ENTER TO CONTINUE.")
    # K = matrix.sum(axis=0)
    return K


def get_fwhm(hist, age):  # TODO: Ezen még lehetne javítani
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


def estimate_age(ego):
    K = make_histogram(ego)

    peak_age = []
    peak_v = []
    for x in range(1, 80):
        if K[x] > K[x - 1] and K[x] > K[x + 1]:
            peak_age.append(x)
            h = K[x]
            w = get_fwhm(K, x)
            v = h / w
            # if x-1 in range(1, 80) and x-2 in range(1, 80) and x+1 in range(1, 80) and x+2 in range(1, 80):
            #    v = (K[x]-K[x-1]) + (K[x]-K[x+1]) + (K[x]-K[x+2]) + (K[x]-K[x-2])
            peak_v.append(v)

    if len(peak_v) == 0:
        # print(ego) TODO:
        return -1
    best_peak_index = peak_v.index(max(peak_v))  # TODO: ha nem talál csúcsot, akkor mit tegyen? 104840 - üres fájl
    estimated_age = peak_age[best_peak_index]
    # estimated_age = min(peak_age, key=lambda x: abs(x - allAges[ego])) # 50.7%-ig megy így # TODO: csúcskiválasztás javítása
    return estimated_age


files = listdir('60-69')
for filename in files:
    if '_out.dot' in filename:
        ego = int(''.join(list(filter(str.isdigit, filename))))
        egos.append(ego)


def read_groups_from_dots():
    groups_of_ego = []
    for ego in egos:
        groups_of_ego.append([ego, read_groups(ego)])
    return dict(groups_of_ego)


def make_smooth_age_distr():
    n = [0]*91
    index = list(range(10, 101))
    N = []
    values = list(allAges.values())
    for i in index:
        N.append(values.count(i))
    for i in range(91):
        for j in range(91):
            n[i] += gauss(i - j)*N[j]   # (i+10) - (j+10) = i-j
    return n


def estimate_all_ages():
    # dev = 0  # Szórás
    counter = 0
    pm2 = 0
    # shuffle(class_egos)
    for ego in class_egos:
        estimated_age = estimate_age(ego)
        if estimated_age == -1:
            continue
        diff = pow((estimated_age - ego.realAge), 2)
        # dev += diff
        if diff <= 4:
            pm2 += 1
        counter += 1
        # if counter % 100 == 0:
        #    print(str(counter) + '/11000')

    # dev /= len(egos)
    # dev = math.sqrt(dev)
    # print('deviation: ' + str(dev))
    # print('+-2: ' + str(pm2 / len(egos)))
    return pm2/len(class_egos)


def calc_derivative(x, n):
    grad = np.zeros(n)
    f_0 = estimate_all_ages()
    print("fitness: " + str(f_0))
    dx = 1
    for dim in range(n):
        x[dim] += dx
        grad[dim] = (estimate_all_ages() - f_0)/dx
        x[dim] -= dx
    return grad

if __name__ == '__main__':
    # Betöltjük a dot fájlokat, amikben a csoportok vannak
    groups_of_ego = load_obj('groups_of_ego')
    # groups_of_ego = read_groups_from_dots()
    # Betöltjük az összes ember korát
    allAges = load_obj('allAges')
    # allAges = read_ages()
    # smoothedAgeDistr = load_obj('smoothedAgeDistr')

    for ego in egos:
        class_ego = Ego(ego, allAges[ego])
        groups = []
        for g in groups_of_ego[ego]:
            ages = []
            for id in g:
                ages.append(allAges[id])
            class_group = Group(ages)
            class_group.calculate_stats()
            if class_group.groupSize != 0: # Ha nem ismerjük egyik ember korát sem a csoportban, akkor a csoportot nem
                groups.append(class_group) # vesszük figyelembe
        class_ego.groups = groups
        if not groups:  # is_empty nincs egy csoportja sem
            continue
        class_egos.append(class_ego)

    start = time.time()
    print(estimate_all_ages())
    end = time.time()
    print(end-start)

    """
    # SA
    opts = np.array([1,  1,  1,   1,  1,  1,
                     1, 1,  3])
    # opts = np.array([1, 1, 1, 1, 1, 1, 1, 1, 4])  # opt, mint optimal, de az algoritmus végén lesz (/lehet) optimális
    E = estimate_all_ages()

    prev_opts = opts
    prev_E = E

    T = 0.001
    while T > 0:
        d_opts = np.array([uniform(-1, 1), uniform(-1, 1), uniform(-1, 1), uniform(-1, 1), uniform(-1, 1), uniform(-1, 1),
                         uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)])
        d_opts = d_opts/np.linalg.norm(opts)
        opts = opts + d_opts
        E = estimate_all_ages()
        print(opts)
        print("E = " + str(E))
        if E > prev_E:
            prev_opts = opts
            prev_E = E
            print("fel")
        else:
            p = math.exp((E-prev_E)/T)
            if uniform(0, 1) <= p:
                prev_opts = opts
                prev_E = E
                print("le " + str(p))
            else:
                opts = opts - d_opts
                print("újra " + str(p))
                # pass # nem csinál semmit, újra random
        print("T = " + str(T) + "\n")
        T -= 0.000001
    """

    """
    # Gradiens módszer - wikipédia 1 D
    opt_sigma = 4.7     # opt, mint optimal, de az algoritmus végén lesz (/lehet) optimális
    dx = 2
    gamma = 10

    f1 = estimate_all_ages()
    prev_opt_sigma = opt_sigma
    opt_sigma += dx
    f2 = estimate_all_ages()
    derivative = (f2 - f1) / dx
    opt_sigma += derivative * gamma
    prev_val = f1
    prev_der = derivative

    precision = 1
    while precision >= 0.00001:
        f1 = estimate_all_ages()
        prev_opt_sigma = opt_sigma
        opt_sigma += dx
        f2 = estimate_all_ages()
        derivative = (f2 - f1)/dx
        print('derivative = ', + str(derivative))
        opt_sigma += derivative*gamma
        # gamma = ((opt_sigma-dx) - prev_opt_sigma)/(derivative - prev_der)
        precision = abs(prev_val - f1)
        prev_val = f1
        prev_der = derivative

    print(precision)
    print(f1)
    print(opt_sigma)

    percents = []
    for i in numpy.linspace(1, 10, 100):
        opt_sigma = i
        percents.append(estimate_all_ages())

    plt.plot(numpy.linspace(1, 10, 100), percents)
    plt.show()
    """

    """
    # Gradiens módszer - wikipédia több Dim
    opts = np.array([1, 1, 1, 1, 1, 1, 1, 1, 4])     # opt, mint optimal, de az algoritmus végén lesz (/lehet) optimális
                          # sorrend: 1/(s + sigma)-ból az s, optimális group size, opt group size sigmája
    gamma = 100
    # fitness: 0.37551606533835935 - csak sigma súly
    # grad = [8.97504936e-05   3.59001975e-04 - 1.79500987e-04]
    # opts = [2.24914737  4.24753186 - 0.88924789]

    grad = calc_derivative(opts, 9)
    prev_opts = opts
    prev_grad = grad
    opts = opts + grad * gamma

    counter = 0
    while np.linalg.norm(grad) >= 0.00001 or counter > 1000:
        start = time.clock()
        grad = calc_derivative(opts, 9)
        # gamma = np.inner((opts - prev_opts), (grad - prev_grad)) / np.linalg.norm(grad - prev_grad)**2
        prev_opts = opts
        prev_grad = grad
        print('grad = ' + str(grad))
        print("opts = " + str(opts))
        print("gamma: " + str(gamma))
        opts = opts + grad * gamma
        counter += 1
        print("time:" + str(time.clock() - start) + "\n")

    print(counter)
    print(estimate_all_ages())
    print(opts)
    """

    """
    percents = []
    for i in np.linspace(1, 10, 100):
        opt_sigma = i
        percents.append(estimate_all_ages())

    plt.plot(np.linspace(1, 10, 100), percents)
    plt.show()
    """
# print(estimate_age(4))
# x = list(range(81))
# plt.plot(x, K)
# plt.show()
