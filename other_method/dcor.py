#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Computes the distance correlation between two matrices. For more detail,
please refer to https://en.wikipedia.org/wiki/Distance_correlation
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform


def dcov(X, Y):
    """Computes the distance covariance between matrices X and Y.

    Args:
        X (np.ndarray): multidimensional array of numbers
        Y (np.ndaaray): multidimensional array of numbers

    Returns:
        (float): the distance covariance between X and Y
    """
    n = X.shape[0]
    XY = np.multiply(X, Y)
    cov = np.sqrt(XY.sum()) / n
    return cov


def dvar(X):
    """Computes the distance variance of a matrix X.

    Args:
        X (np.ndarray): multidimensional array of numbers

    Returns:
        (float): the distance variance of X
    """
    return np.sqrt(np.sum(X ** 2 / X.shape[0] ** 2))


def cent_dist(X):
    """Computes pairwise euclidean distance between rows of X and centers each
    cell of the distance matrix with row mean, column mean, and grand mean.

    Args:
        X (np.ndarray): multidimensional array of numbers

    Returns:
        (np.ndarray): doubly centered distance matrix of X
    """
    M = squareform(pdist(X))    # distance matrix
    rmean = M.mean(axis=1)
    cmean = M.mean(axis=0)
    gmean = rmean.mean()
    R = np.tile(rmean, (M.shape[0], 1)).transpose()
    C = np.tile(cmean, (M.shape[1], 1))
    G = np.tile(gmean, M.shape)
    CM = M - R - C + G
    return CM


def dcor(X, Y):
    """Computes the distance correlation between two matrices X and Y.

    X and Y must have the same number of rows.

    >>> X = np.matrix('1;2;3;4;5')
    >>> Y = np.matrix('1;2;9;4;4')
    >>> dcor(X, Y)
    0.76267624241686649

    Args:
        X (np.ndarray): multidimensional array of numbers
        Y (np.ndarray): multidimensional array of numbers

    Returns:
        (float, float, float, float): (dCorr(X, Y), dCov(X, Y), dVar(X),
        dVar(Y))
    """
    assert X.shape[0] == Y.shape[0]

    A = cent_dist(X)
    B = cent_dist(Y)

    dcov_AB = dcov(A, B)
    dvar_A = dvar(A)
    dvar_B = dvar(B)

    dcor = 0.0
    if dvar_A > 0.0 and dvar_B > 0.0:
        dcor = dcov_AB / np.sqrt(dvar_A * dvar_B)

    return dcor, dcov_AB, dvar_A, dvar_B


if __name__ == "__main__":
    X = np.matrix('1;2;3;4;5')
    Y = np.matrix('1;2;9;4;4')
    print(dcor(X, Y))
    # print(dcor(np.matrix('1 7 3; 8 2 9; 1 2 7'), np.matrix('9 6; 2 3; 1 8')))
