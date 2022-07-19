"""This module assess the performance of various discrete causal inference
method on systhetic cause-effect pairs in fixed dimention |X| and |Y|
"""

from __future__ import division
import math
import random
import sys
import time
import numpy as np
import pandas as pd

from src import ndm

random.seed(0)
np.random.seed(0)

def map_randomly(dom_f, img_f):
    f = dict((x, random.choice(img_f)) for x in dom_f)
    # ensure that f is not a constant function
    if len(set(f.values())) == 1:
        f = map_randomly(dom_f, img_f)
    assert len(set(f.values())) != 1
    return f


def generate_X(size: int):
    # generate X from multinomial disctibution
    p_nums = [random.randint(1, 100) for _ in range(20)]
    p_vals = [v / sum(p_nums) for v in p_nums]
    X = np.random.multinomial(size, p_vals, size=1)[0].tolist()
    X = [[i] * f for i, f in enumerate(X)]
    X = [j for sublist in X for j in sublist]
    return X
    

def map_randomly(dom_f, img_f):
    f = dict((x, random.choice(img_f)) for x in dom_f)
    # ensure that f is not a constant function
    if len(set(f.values())) == 1:
        f = map_randomly(dom_f, img_f)
    assert len(set(f.values())) != 1
    return f

def test_accuracy_4types:
    nsim = 100
    sample_sizes = [100, 1000, 10000]
    img_f = range(20)
    dom_f = range(20)

    print("%18s%10s" % ("X", "NDM"))
    print("-" * 80)
    sys.stdout.flush()
    fp = open("results/fixed-dim-acc-nsize.dat", "w")
    fp.write("%s\t%s\n" % ("size", "ndm"))

    for sample_size in sample_sizes:
        nc_ndm = 0
        for i in range(nsim):
            X = generate_X(sample_size)
            f = map_randomly(dom_f, img_f)
            N = generate_X(sample_size)
            Y = [(f[X[i]] + N[i]) % 20 for i in range(sample_size)]

            assert len(X) == len(Y) == len(N)

            sys.stdout.write("\r{}/{}".format(i+1, nsim))
            sys.stdout.flush()
            ndm_score = ndm(X, Y)
            ndm_score.sort(key=lambda x: x[0])
            ndm_score = ndm_score[0][1]
            if ndm_score == "to":
                nc_ndm += 1

        acc_ndm = nc_ndm * 100 / nsim
        print("%18s%10.2f" % (sample_size, acc_ndm))
        sys.stdout.flush()
        fp.write(
            "%s\t%.2f\n" % (sample_size, acc_ndm)
        )

    print("-"*80)
    sys.stdout.flush()
    fp.close()


if __name__ == "__main__":
    main()
