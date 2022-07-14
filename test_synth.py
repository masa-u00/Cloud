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

from src import ndm


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

def map_randomly(dom_f, img_f):
    f = dict((x, random.choice(img_f)) for x in dom_f)
    # ensure that f is not a constant function
    if len(set(f.values())) == 1:
        f = map_randomly(dom_f, img_f)
    assert len(set(f.values())) != 1
    return f


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

if __name__ == "__main__":
    nsim = 1000
    sample_size = 1000
    img_f = range(-7, 8)
    srcsX = ["uniform", "binomial", "negativeBinomial",
         "geometric", "hypergeometric", "poisson", "multinomial"]
    print("-" * 80)
    print("%18s%10s" % ("X", "NDM"))
    print("-" * 80)
    sys.stdout.flush()
    fp = open("results/acc-dtype.dat", "w")
    fp.write("%s\t%s\n" % ("dtype", "ndm"))
    for srcX in srcsX:
        nsamples = 0
        nc_ndm = 0
        while nsamples < nsim:
            X = generate_X(srcX, sample_size)
            dom_f = list(set(X))
            f = map_randomly(dom_f, img_f)
            N = generate_additive_N(sample_size)
            Y = [f[X[i]] + N[i] for i in range(sample_size)]

            assert len(X) == len(Y) == len(N)

            if not identifiable(dom_f, f, N):
                continue

            nsamples += 1
            ndm_score = ndm(X, Y)
            ndm_score.sort(key=lambda x: x[0])
            ndm_score = ndm_score[0][1]

            if ndm_score == "to":
                nc_ndm += 1

        assert nsamples == nsim

        acc_ndm = nc_ndm * 100 / nsim
        print("%18s%10.2f" % (srcX, acc_ndm))
        sys.stdout.flush()
        fp.write(
            "%s\t%.2f\n" % (srcX, acc_ndm)
        )
    print("-" * 80)
    sys.stdout.flush()
    fp.close()
