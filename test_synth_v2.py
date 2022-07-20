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
import json
from collections import defaultdict

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


def generate_sequence(dom_size: int, size: int):
    # generate X from multinomial disctibution
    p_nums = [np.random.random() for _ in range(dom_size)]
    p_vals = [v / sum(p_nums) for v in p_nums]
    X = np.random.choice(a=range(dom_size), p=p_vals, size=size)
    return X
    

def map_randomly(dom_f, img_f):
    f = dict((x, random.choice(img_f)) for x in dom_f)
    # ensure that f is not a constant function
    if len(set(f.values())) == 1:
        f = map_randomly(dom_f, img_f)
    assert len(set(f.values())) != 1
    return f

def test_accuracy_4types(args):
    nsim = 1000
    sample_sizes = [100, 1000, 10000, 100000]
    causal_types = ["to", "indep", "confounder"]

    img_f = range(args.img)
    dom_f = range(args.dom)

    print("%18s%18s%10s" % ("causal type", "sample_size", "NDM"))
    print("-" * 80)
    sys.stdout.flush()
    fp = open("results/fixed-dim-acc-nsize.dat", "w")
    fp.write("%s\t%s\t%s\n" % ("causal_type", "size", "ndm"))

    dd = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for causal_type in causal_types:
        for sample_size in sample_sizes:
            nc_ndm = 0
            for i in range(nsim):
                if causal_type == "to":
                    X = generate_sequence(args.dom, sample_size)
                    f = map_randomly(dom_f, img_f)
                    N = generate_sequence(args.img, sample_size)
                    Y = [(f[x] + n) % args.img for x, n in zip(X, N)]

                elif causal_type == "gets":
                    Y = generate_sequence(args.img, sample_size)
                    g = map_randomly(img_f, dom_f)
                    N = generate_sequence(args.dom, sample_size)
                    X = [(g[y] + n) % args.dom for y, n in zip(Y, N)]

                elif causal_type == "indep":
                    X = generate_sequence(args.dom, sample_size)
                    Y = generate_sequence(args.img, sample_size)

                elif causal_type == "confounder":
                    X = generate_sequence(args.dom, sample_size)
                    Y = generate_sequence(args.img, sample_size)
                    C = generate_sequence(args.confounder, sample_size)
                    C2X = map_randomly(range(args.confounder), range(args.dom))
                    C2Y = map_randomly(range(args.confounder), range(args.img))

                    X = [(x + C2X[c]) % args.dom for x, c in zip(X, C)]
                    #X = [(x + c) % args.dom for x, c in zip(X, C)]
                    #X = (generate_sequence(args.dom, sample_size) + C) % args.dom
                    Y = [(y + C2X[c]) % args.img for y, c in zip(Y, C)]
                    #Y = [(y + c) % args.img for y, c in zip(Y, C)]
                    #Y = (generate_sequence(args.img, sample_size) + C) % args.img

                assert len(X) == len(Y)

                sys.stdout.write("\r{}/{}".format(i+1, nsim))
                sys.stdout.flush()
                ndm_score = ndm(X, Y)
                ndm_score.sort(key=lambda x: x[0])
                ndm_score = ndm_score[0][1]
                dd[causal_type][ndm_score][sample_size] += 1
                if ndm_score == causal_type:
                    nc_ndm += 1

            acc_ndm = nc_ndm * 100 / nsim
            print("%18s%18s%10.2f" % (causal_type, sample_size, acc_ndm))
            sys.stdout.flush()
            fp.write(
                "%s\t%s\t%.2f\n" % (causal_type, sample_size, acc_ndm)
            )
    with open("results/test_acc_4types.json", "w") as f:
        json.dump(dd, f)

    print("-"*80)
    sys.stdout.flush()
    fp.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="systhetic experiment")
    parser.add_argument("--dom", type=int, default=20, help="domain size")
    parser.add_argument("--img", type=int, default=20, help="image size")
    parser.add_argument("--confounder", type=int, default=10, help="domain size of confounder")
    args = parser.parse_args()

    test_accuracy_4types(args)


