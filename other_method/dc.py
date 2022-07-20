#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module implements the work on `Causal Inference on Discrete Data via
Estimating Distance Correlations`. For more detail, please refer to the
manuscript at http://www.mitpressjournals.org/doi/pdf/10.1162/NECO_a_00820
"""
import numpy as np

from dcor import dcor


def dc(X, Y):
    """Computes dCorr(P(X), P(Y|X)), and dCorr(P(Y), P(X|Y)).

    Args:
        X (nested sequence): nested sequence of discrete outcomes
        Y (nested sequence): nested sequence of discrete outcomes

    Returns:
        (float, float): (dCorr(P(X), P(Y|X)), dCorr(P(Y), P(X|Y)))
    """
    assert len(X) == len(Y)
    marg_X, cond_X, marg_Y, cond_Y = distributions(X, Y)
    dXtoY = dcor(marg_X, cond_Y)[0]
    dYtoX = dcor(marg_Y, cond_X)[0]
    return (dXtoY, dYtoX)


def distributions(X, Y):
    """Computes empirical marginal and conditional distributions of X and Y.

    Args:
        X (nested sequence): nested sequence of discrete outcomes
        Y (nested sequence): nested sequence of discrete outcomes

    Returns:
        (sequence, sequence, sequence, sequence): (P(X), P(X|Y), P(Y), P(Y|X)).
        If X has L unique values, and Y has M unique values. The dimension are
        as follows: P(X)=Lx1, P(Y|X)=LxM, P(Y)=Mx1, and P(X|Y)=MxL.
    """
    N = len(X)
    unq_X = set(map(tuple, X))
    unq_Y = set(map(tuple, Y))
    idx = range(N)
    idx_X = dict(zip(unq_X, idx))
    idx_Y = dict(zip(unq_Y, idx))

    freq_XY = np.zeros((len(unq_X), len(unq_Y)))
    for i in range(N):
        ix = idx_X[tuple(X[i])]
        iy = idx_Y[tuple(Y[i])]
        freq_XY[ix, iy] += 1

    freq_X = np.sum(freq_XY, axis=1)[np.newaxis]
    freq_Y = np.sum(freq_XY, axis=0)[np.newaxis]
    marg_X = (freq_X / np.sum(freq_X)).transpose()
    marg_Y = (freq_Y / np.sum(freq_Y)).transpose()

    freqs_X = np.tile(freq_X.transpose(), (1, len(unq_Y)))
    freqs_Y = np.tile(freq_Y, (len(unq_X), 1))
    cond_X = (freq_XY / freqs_X).transpose()
    cond_Y = (freq_XY / freqs_Y)
    return marg_X, cond_X, marg_Y, cond_Y


if __name__ == "__main__":
    X = [[2, 3], [2, 3], [2, 4], [2], [2], [3], [3], [3, 4], [2, 3], [2]]
    Y = [[1], [1], [1], [1], [1], [0], [0], [0], [0], [0]]
    print(dc(X, Y))
