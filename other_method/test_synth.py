#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module assess the performance of various discrete causal inference
methods on synthetic cause-effect pairs.
"""
from __future__ import division
import math
import random
import sys
import time

import numpy as np
import pandas as pd

from anm import anm
from cisc import cisc
from dc import dc
from entropic import entropic
from formatter import to_nested
from measures import ChiSquaredTest, Entropy, StochasticComplexity


def reverse_argsort(sequence):
    indices = range(len(sequence))
    indices.sort(key=sequence.__getitem__, reverse=True)
    return indices


def map_randomly(dom_f, img_f):
    f = dict((x, random.choice(img_f)) for x in dom_f)
    # ensure that f is not a constant function
    if len(set(f.values())) == 1:
        f = map_randomly(dom_f, img_f)
    assert len(set(f.values())) != 1
    return f


def generate_X(type, size):
    if type == "uniform":
        maxX = random.randint(2, 10)
        X = [random.randint(1, maxX) for i in range(size)]
    elif type == "multinomial":
        p_nums = [random.randint(1, 10) for i in range(random.randint(2, 11))]
        p_vals = [v / sum(p_nums) for v in p_nums]
        X = np.random.multinomial(size, p_vals, size=1)[0].tolist()
        X = [[i + 1] * f for i, f in enumerate(X)]
        X = [j for sublist in X for j in sublist]
    elif type == "binomial":
        n = random.randint(1, 40)
        p = random.uniform(0.1, 0.9)
        X = np.random.binomial(n, p, size).tolist()
    elif type == "geometric":
        p = random.uniform(0.1, 0.9)
        X = np.random.geometric(p, size).tolist()
    elif type == "hypergeometric":
        ngood = random.randint(1, 40)
        nbad = random.randint(1, 40)
        nsample = random.randint(1, min(40, ngood + nbad - 1))
        X = np.random.hypergeometric(ngood, nbad, nsample, size).tolist()
    elif type == "poisson":
        rate = random.randint(1, 10)
        X = np.random.poisson(rate, size).tolist()
    elif type == "negativeBinomial":
        n = random.randint(1, 40)
        p = random.uniform(0.1, 0.9)
        X = np.random.negative_binomial(n, p, size).tolist()
    return X


def generate_additive_N(n):
    t = random.randint(1, 7)
    suppN = range(-t, t + 1)
    N = [random.choice(suppN) for i in range(n)]
    return N


def are_disjoint(sets):
    disjoint = True
    union = set()
    for s in sets:
        for x in s:
            if x in union:
                disjoint = False
                break
            union.add(x)
    return disjoint


def identifiable(dom_f, f, N):
    # check if f(x) + supp N are disjoint for x in domain of f
    supp_N = set(N)
    decomps = []
    for x in dom_f:
        u = f[x]
        supp_addition = set([u + n for n in supp_N])
        decomps.append(supp_addition)
    non_overlapping_noise = are_disjoint(decomps)
    return not non_overlapping_noise


def calibrate_dr_alpha():
    print("testing various critical thresholds for DR")
    sys.stdout.flush()
    nsim = 1000
    size = 1000
    level = 0.01
    suppfX = range(-7, 8)
    srcX = "geometric"
    levels = [0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001]
    print("-" * 24)
    print("%8s%15s" % ("ALPHA", "ACCURACY(DR)"))
    print("-" * 24)
    for level in levels:
        nsamples = 0
        ncorrect = 0
        while nsamples < nsim:
            X = generate_X(srcX, size)
            suppX = list(set(X))
            f = map_randomly(suppX, suppfX)
            N = generate_additive_N(size)
            Y = [f[X[i]] + N[i] for i in range(size)]

            if not identifiable(suppX, f, N):
                continue

            nsamples += 1
            dr_score = anm(X, Y, ChiSquaredTest, 100)
            ncorrect += int(dr_score[0] > level and dr_score[1] < level)

        assert nsamples == nsim
        acc = ncorrect * 100 / nsim
        print("%8.3f%15.2f" % (level, acc))
        sys.stdout.flush()
    print("-" * 24)


def test_accuracy_data_type():
    nsim = 1000
    sample_size = 1000
    level = 0.05
    img_f = range(-7, 8)
    srcsX = ["uniform", "binomial", "negativeBinomial",
             "geometric", "hypergeometric", "poisson", "multinomial"]
    print("-" * 80)
    print("%18s%10s%10s%10s%10s%10s%10s" %
          ("X", "DC", "ENT", "DR", "CISC", "ACID", "CRISP"))
    print("-" * 80)
    sys.stdout.flush()
    fp = open("results/acc-dtype.dat", "w")
    fp.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" %
             ("dtype", "dc", "ent", "dr", "cisc", "acid", "crisp"))
    for srcX in srcsX:
        nsamples = 0
        nc_dc, nc_ent, nc_dr, nc_cisc, nc_acid, nc_crisp = 0, 0, 0, 0, 0, 0
        while nsamples < nsim:
            X = generate_X(srcX, sample_size)
            dom_f = list(set(X))
            f = map_randomly(dom_f, img_f)
            N = generate_additive_N(sample_size)
            Y = [f[X[i]] + N[i] for i in range(sample_size)]

            assert len(X) == len(Y) == len(N)

#            if not identifiable(dom_f, f, N):
#                continue

            nsamples += 1
            dc_score = dc(to_nested(X), to_nested(Y))
            ent_score = entropic(pd.DataFrame(np.column_stack((X, Y))))
            dr_score = anm(X, Y, ChiSquaredTest, 100)
            cisc_score = cisc(X, Y)
            acid_score = anm(X, Y, Entropy, 100)
            crisp_score = anm(X, Y, StochasticComplexity, 100, True)
            # dc_score = (0, 0)
            # dr_score = (0, 0)
            # cisc_score = (0, 0)
            # acid_score = (0, 0)
            # crisp_score = (0, 0)

            nc_dc += int(dc_score[0] < dc_score[1])
            nc_ent += int(ent_score[0] < ent_score[1])
            nc_dr += int(dr_score[0] > level and dr_score[1] < level)
            nc_cisc += int(cisc_score[0] < cisc_score[1])
            nc_acid += int(acid_score[0] < acid_score[1])
            nc_crisp += int(crisp_score[0] < crisp_score[1])

        assert nsamples == nsim

        acc_dc = nc_dc * 100 / nsim
        acc_ent = nc_ent * 100 / nsim
        acc_dr = nc_dr * 100 / nsim
        acc_cisc = nc_cisc * 100 / nsim
        acc_acid = nc_acid * 100 / nsim
        acc_crisp = nc_crisp * 100 / nsim
        print("%18s%10.2f%10.2f%10.2f%10.2f%10.2f%10.2f" %
              (srcX, acc_dc, acc_ent, acc_dr, acc_cisc, acc_acid, acc_crisp))
        sys.stdout.flush()
        fp.write(
            "%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n" %
            (srcX, acc_dc, acc_ent, acc_dr, acc_cisc, acc_acid, acc_crisp)
        )
    print("-" * 80)
    sys.stdout.flush()
    fp.close()


def test_accuracy_sample_size():
    nsim = 1000
    level = 0.05
    sizes = [100, 400, 700, 1000, 1300, 1600, 2000, 10000]
    suppfX = range(-7, 8)
    srcX = "geometric"

    fp = open("results/acc-size.dat", "w")
    fp.write("%s\t%s\t%s\t%s\t%s\t%s\n" %
             ("size", "dc", "dr", "cisc", "acid", "entropic"))
    print("%s\t%s\t%s\t%s\t%s\t%s" %
          ("size", "dc", "dr", "cisc", "acid", "entropic"))
    sys.stdout.flush()
    for k, size in enumerate(sizes):
        nsamples = 0
        nc_dc, nc_ent, nc_dr, nc_cisc, nc_acid, nc_crisp = 0, 0, 0, 0, 0, 0
        while nsamples < nsim:
            X = generate_X(srcX, size)
            suppX = list(set(X))
            f = map_randomly(suppX, suppfX)
            N = generate_additive_N(size)
            Y = [f[X[i]] + N[i] for i in range(size)]

            if not identifiable(suppX, f, N):
                continue

            nsamples += 1
            dc_score = dc(to_nested(X), to_nested(Y))
            ent_score = entropic(pd.DataFrame(np.column_stack((X, Y))))
            dr_score = anm(X, Y, ChiSquaredTest, 100)
            cisc_score = cisc(X, Y)
            acid_score = anm(X, Y, Entropy, 100)
            crisp_score = anm(X, Y, StochasticComplexity, 100, True)

            nc_dc += int(dc_score[0] < dc_score[1])
            nc_ent += int(ent_score[0] < ent_score[1])
            nc_dr += int(dr_score[0] > level and dr_score[1] < level)
            nc_cisc += int(cisc_score[0] < cisc_score[1])
            nc_acid += int(acid_score[0] < acid_score[1])
            nc_crisp += int(crisp_score[0] < crisp_score[1])

        assert nsamples == nsim

        acc_dc = nc_dc * 100 / nsim
        acc_ent = nc_ent * 100 / nsim
        acc_dr = nc_dr * 100 / nsim
        acc_cisc = nc_cisc * 100 / nsim
        acc_acid = nc_acid * 100 / nsim
        acc_crisp = nc_crisp * 100 / nsim

        print("%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f" %
              (size, acc_dc, acc_ent, acc_dr, acc_cisc, acc_acid, acc_crisp))
        sys.stdout.flush()
        fp.write(
            "%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n" %
            (size, acc_dc, acc_ent, acc_dr, acc_cisc, acc_acid, acc_crisp))
    fp.close()


def test_hypercompression():
    m = 100
    size = 350
    alpha = 0.01
    suppfX = range(-7, 8)
    srcX = "geometric"

    fp = open("results/no-hypercompression.dat", "w")
    diffs = []
    decisions = []  # 1=correct, -1=incorrect, 0=wrong
    nsamples = 0
    while nsamples < m:
        X = generate_X(srcX, size)
        suppX = list(set(X))
        f = map_randomly(suppX, suppfX)
        N = generate_additive_N(size)
        Y = [f[X[i]] + N[i] for i in range(size)]

        if not identifiable(suppX, f, N):
            continue

        nsamples += 1
        acid_score = anm(X, Y, Entropy, 100)
        diff = size * abs(acid_score[0] - acid_score[1])

        if acid_score[0] < acid_score[1]:
            decision = 1
        elif acid_score[0] > acid_score[1]:
            decision = -1
        else:
            continue

        diffs.append(int(diff))
        decisions.append(decision)

    sorted_diffs_indices = reverse_argsort(diffs)
    diffs = [diffs[idx] for idx in sorted_diffs_indices]
    decisions = [decisions[idx] for idx in sorted_diffs_indices]

    # flags for coloring
    # correct, significant = 1
    # correct, insignificant = 2
    # incorrect, significant = 3
    # incorrect, insignificant = 4

    fp.write("sn\tdiff\tsig\tdec\tcolor\n")  # header
    for k, diff in enumerate(diffs, 1):
        log_p_value = -diff
        bh_stat = k * alpha / m
        log_bh_stat = math.log(bh_stat, 2)

        if log_bh_stat < log_p_value:
            significant = 0
            if decisions[k - 1] == 1:
                color = 2
            else:
                color = 4
        else:
            significant = 1
            if decisions[k - 1] == 1:
                color = 1
            elif decisions[k - 1] == -1:
                color = 3

        fp.write("%i\t%d\t%d\t%d\t%d\n" %
                 (k, diff, significant, decisions[k - 1], color))
        # if log_bh_stat < log_p_value:
        #     # reject: not significant
        #     fp.write("%i\t%d\t%d\t%d\n" % (k, diff, 0, decisions[k - 1]))
        #     print k, diff, log_bh_stat, log_p_value, 0, decisions[k - 1]
        # else:
        #     fp.write("%i\t%d\t%d\t%d\n" %
        #              (k, diff, 1, decisions[k - 1]))  # accept: significant
        #     print k, diff, 1, decisions[k - 1]

        # fp.write("%i\t%d\n" % (k, diff))
    fp.close()


def test_domain_runtime_acid():
    nloop = 5
    size = 10000
    supps = [20, 40, 80, 160]
    for supp in supps:
        pool = range(supp)
        X = [random.choice(pool) for i in range(size)]
        Y = [random.choice(pool) for i in range(size)]
        acid_t = 0
        for i in range(nloop):
            tstart = time.time()
            anm(X, Y, Entropy, 100)
            tend = time.time()
            acid_t += tend - tstart
        print(supp, acid_t / nloop)
        sys.stdout.flush()


def test_size_runtime_acid():
    pool = range(-7, 8)
    nloop = 5
    sizes = [10000, 20000, 40000, 80000, 160000, 320000, 640000, 1280000]
    for size in sizes:
        X = [random.choice(pool) for i in range(size)]
        Y = [random.choice(pool) for i in range(size)]
        acid_t = 0
        for i in range(nloop):
            tstart = time.time()
            anm(X, Y, Entropy, 100)
            tend = time.time()
            acid_t += tend - tstart
        print(size, acid_t / nloop)
        sys.stdout.flush()


if __name__ == "__main__":
    # calibrate_dr_alpha()
    test_accuracy_data_type()
    # test_accuracy_sample_size()
    # test_hypercompression()
    # test_size_runtime_acid()
    # test_domain_runtime_acid()
