#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module implements the paper titled `MDL for Causal Inference on
Discrete Data`. For more detail, please refer to the manuscript at
http://people.mpi-inf.mpg.de/~kbudhath/manuscript/cisc.pdf
"""
from formatter import stratify
from sc import sc


def cisc(X, Y, plain=False):
    """Computes the total stochastic complexity from X to Y and vice versa.

    Args:
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes
        plain (bool): whether to compute the plain conditional stochastic
            complexity or not. If not provided, we compute the weighted one.

    Returns:
        (float, float): the total multinomial stochastic complexity of X and Y
            in the direction from X to Y, and vice versa.
    """
    assert len(X) == len(Y)

    n = len(X)

    scX = sc(X)
    scY = sc(Y)

    YgX = stratify(X, Y)
    XgY = stratify(Y, X)

    domX = YgX.keys()
    domY = XgY.keys()

    ndomX = len(domX)
    ndomY = len(domY)

    if plain:
        scYgX = sum(sc(Yp, ndomY) for Yp in YgX.values())
        scXgY = sum(sc(Xp, ndomX) for Xp in XgY.values())
    else:
        scYgX = sum(len(Yp) / n * sc(Yp, ndomY) for Yp in YgX.values())
        scXgY = sum(len(Xp) / n * sc(Xp, ndomX) for Xp in XgY.values())

    ciscXtoY = scX + scYgX
    ciscYtoX = scY + scXgY

    return (ciscXtoY, ciscYtoX)


if __name__ == "__main__":
    import random
    n = 100
    X = [random.randint(0, 10) for i in range(n)]
    Y = [random.randint(0, 10) for i in range(n)]
    print(cisc(X, Y))
