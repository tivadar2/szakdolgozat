# coding=utf-8

import math
import pickle
import matplotlib.pyplot as plt
from os import listdir
import numpy

# Globális változók
allAges = {}
groups_of_ego = {}
egos = []
smoothedAgeDistr = []

# Beállítáok
opt_sigma = 1
opt_gSize = 5


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


def make_histogram(ego):
    groups = groups_of_ego[ego]
    K = [0] * 81
    for g in groups:
        group_ages = []
        for ID in g:
            age = allAges[ID]
            if age == 999 or age == 555: # TODO: ez mi?
                continue
            group_ages.append(age)
        if len(group_ages) == 0:  # Ha mindegyik ember a csoportban 999 éves # TODO: működik?
            continue
        groupSize = len(group_ages)
        avg = sum(group_ages)/groupSize
        sigma = 0
        for age in group_ages:
            sigma += (avg - age)**2
        sigma = math.sqrt(sigma/groupSize)
        for a in range(10, 81):
            # K[a] += gauss(a - avg)/smoothedAgeDistr[int(avg+0.5)-10]
            # K[a] += gauss(a - avg)

            # *gauss(groupSize - 3)
            K[a] += gauss(a - avg)/(opt_sigma + sigma)     # TODO:
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
            peak_v.append(v)

    if len(peak_v) == 0:
        # print(ego) TODO:
        return -1
    best_peak_index = peak_v.index(max(peak_v))  # TODO: ha nem talál csúcsot, akkor mit tegyen? 104840 - üres fájl
    estimated_age = peak_age[best_peak_index]
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
    dev = 0  # Szórás
    counter = 0
    pm2 = 0
    for ego in egos:
        estimated_age = estimate_age(ego)
        if estimated_age == -1:
            continue
        real_age = allAges[ego]
        diff = pow((estimated_age - real_age), 2)
        dev += diff
        if diff <= 4:
            pm2 += 1
        counter += 1
        if counter % 100 == 0:
            print(str(counter) + '/11000')

    dev /= len(egos)
    dev = math.sqrt(dev)
    # print('deviation: ' + str(dev))
    # print('+-2: ' + str(pm2 / len(egos)))
    return pm2/len(egos)

if __name__ == '__main__':
    # Betöltjük a dot fájlokat, amikben a csoportok vannak
    groups_of_ego = load_obj('groups_of_ego')
    # groups_of_ego = read_groups_from_dots()
    # Betöltjük az összes ember korát
    allAges = load_obj('allAges')
    # allAges = read_ages()
    smoothedAgeDistr = load_obj('smoothedAgeDistr')


    # Gradiens módszer
    opt_sigma = 4.7
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
    """
    percents = []
    for i in numpy.linspace(1, 10, 100):
        opt_sigma = i
        percents.append(estimate_all_ages())

    plt.plot(numpy.linspace(1, 10, 100), percents)
    plt.show()
    """
# print(estimate_age(4))
# x = list(range(81))
# plt.plot(x, K)
# plt.show()
