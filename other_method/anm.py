#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module implements the paper titled `Accurate Causal Inference on
Discrete Data`. We can also compute the total information content in the
sample by encoding the function and using the stochastic complexity on top of
regression model. For more detail, please refer to the manuscript at
http://people.mpi-inf.mpg.de/~kbudhath/manuscript/acid.pdf
"""
from collections import Counter
from math import log
import sys

from formatter import stratify
from measures import DependenceMeasure, DMType


def choose(n, k):
    """Computes the binomial coefficient `n choose k`.
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


def univ_enc(n):
    """Computes the universal code length of the given integer.
    Reference: J. Rissanen. A Universal Prior for Integers and Estimation by
    Minimum Description Length. Annals of Statistics 11(2) pp.416-431, 1983.
    """
    ucl = log(2.86504, 2)
    previous = n
    while True:
        previous = log(previous, 2)
        if previous < 1.0:
            break
        ucl += previous
    return ucl


def encode_func(f):
    """Encodes the function by enumerating the set of all possible functions.

    Args:
        ndom (int): number of elements in the domain of the function
        nimg (int): number of elements in the image of the function

    Returns:
        (float): encoded size of the function
    """
    # nones = len(set(f.values()))
    # return univ_enc(nones) + log(choose(ndom * nimg, nones), 2)
    ndom = len(f.keys())
    nimg = len(set(f.values()))
    return univ_enc(ndom) + univ_enc(nimg) + log(ndom ** nimg, 2)


def map_to_majority(X, Y):
    """Creates a function that maps y to the frequently co-occuring x.

    Args:
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes

    Returns:
        (dict): map from Y-values to frequently co-occuring X-values
    """
    f = dict()
    Y_grps = stratify(X, Y)
    for x, Ys in Y_grps.items():
        frequent_y, _ = Counter(Ys).most_common(1)[0]
        f[x] = frequent_y
    return f


def regress(X, Y, dep_measure, max_niterations, enc_func=False):
    """Performs discrete regression with Y as a dependent variable and X as
    an independent variable.

    Args:
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes
        dep_measure (DependenceMeasure): subclass of DependenceMeasure
        max_niterations (int): maximum number of iterations
        enc_func (bool): whether to encode the function or not

    Returns:
        (float): p-value (or information content) after fitting ANM from X->Y
    """
    # todo: make it work with chi-squared test of independence or G^2 test
    supp_X = list(set(X))
    supp_Y = list(set(Y))
    f = map_to_majority(X, Y)

    pair = list(zip(X, Y))
    res = [y - f[x] for x, y in pair]
    cur_res_inf = dep_measure.measure(res, X)

    j = 0
    minimized = True
    while j < max_niterations and minimized:
        minimized = False

        for x_to_map in supp_X:
            best_res_inf = sys.float_info.max
            best_y = None

            for cand_y in supp_Y:
                if cand_y == f[x_to_map]:
                    continue

                res = [y - f[x] if x != x_to_map else y -
                       cand_y for x, y in pair]
                res_inf = dep_measure.measure(res, X)

                if res_inf < best_res_inf:
                    best_res_inf = res_inf
                    best_y = cand_y

            if best_res_inf < cur_res_inf:
                cur_res_inf = best_res_inf
                f[x_to_map] = best_y
                minimized = True
        j += 1

    if dep_measure.type == DMType.INFO and not enc_func:
        return dep_measure.measure(X) + cur_res_inf
    elif dep_measure.type == DMType.INFO and enc_func:
        return dep_measure.measure(X) + encode_func(f) + cur_res_inf
    else:
        _, p_value = dep_measure.nhst([y - f[x] for x, y in pair], X)
        return p_value


def anm(X, Y, dep_measure, max_niterations=1000, enc_func=False):
    """Fits the Additive Noise Model from X to Y and vice versa.

    Args:
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes
        dep_measure (DependenceMeasure): subclass of DependenceMeasure
        max_niterations (int): maximum number of iterations
        enc_func (bool): whether to encode the function or not

    Returns:
        (float, float): p-value (or information content) after fitting ANM
        from X->Y and vice versa.
    """
    assert issubclass(dep_measure, DependenceMeasure), "dependence measure "\
        "must be a subclass of DependenceMeasure abstract class"
    xtoy = regress(X, Y, dep_measure, max_niterations, enc_func)
    ytox = regress(Y, X, dep_measure, max_niterations, enc_func)
    return (xtoy, ytox)


if __name__ == "__main__":
    import numpy as np
    from measures import Entropy, StochasticComplexity, ChiSquaredTest

    X = np.random.choice([1, 2, 4, -1], 1000)
    Y = np.random.choice([-2, -1, 0, 1, 2], 1000)

    print(anm(X, Y, Entropy))
    print(anm(X, Y, StochasticComplexity))
    print(anm(X, Y, StochasticComplexity, enc_func=True))
    print(anm(X, Y, ChiSquaredTest))
