#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module implements the linear algorithm for computing the stochastic
complexity of a discrete sequence relative to a parametric family of
multinomial distributions. For more detail, please refer to
http://pgm08.cs.aau.dk/Papers/31_Paper.pdf
"""
from __future__ import division
from collections import Counter
from math import ceil, log, sqrt


def log2(n):
    return log(n or 1, 2)


def model_cost(ndistinct_vals, n):
    """Computes the logarithm of the normalising term of multinomial
    stochastic complexity.

    Args:
        ndistinct_vals (int): number of distinct values of a multinomial r.v.
        n (int): number of trials

    Returns:
        float: the model cost of the parametric family of multinomials
    """
    total = 1.0
    b = 1.0
    d = 10

    bound = int(ceil(2 + sqrt(2 * n * d * log(10))))  # using equation (38)
    for k in range(1, bound + 1):
        b = (n - k + 1) / n * b
        total += b

    log_old_sum = log2(1.0)
    log_total = log2(total)
    log_n = log2(n)
    for j in range(3, ndistinct_vals + 1):
        log_x = log_n + log_old_sum - log_total - log2(j - 2)
        x = 2 ** log_x
        # log_one_plus_x = (x + 8 * x / (2 + x) + x / (1 + x)) / 6
        log_one_plus_x = log2(1 + x)
        # one_plus_x = 1 + n * 2 ** log_old_sum / (2 ** log_total * (j - 2))
        # log_one_plus_x = log2(one_plus_x)
        log_new_sum = log_total + log_one_plus_x
        log_old_sum = log_total
        log_total = log_new_sum
        # print log_total,

    if ndistinct_vals == 1:
        log_total = log2(1.0)

    return log_total


def sc(X, ndistinct_vals=None):
    """Computes the stochastic complexity of a discrete sequence.

    Args:
        X (sequence): sequence of discrete outcomes
        ndistinct_vals (int): number of distinct values of the multinomial
            r.v. X. If not provided, we take it directly from X.

    Returns:
        float: the multinomial stochastic complexity of X
    """
    freqs = Counter(X)
    n = len(X)
    ndistinct_vals = ndistinct_vals or len(freqs)
    data_cost = 0.0
    for freq in freqs.values():
        data_cost += freq * (log2(n) - log2(freq))
    return data_cost + model_cost(ndistinct_vals, n)


if __name__ == "__main__":
    print(sc([1, 2, 3, 2, 1, 2]))
