#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module implements the Shannon entropy of a discrete sequence.
"""
from collections import Counter
from math import log


def entropy(X):
    """Compute the entropy of a message.

    Args:
        X (sequence): a sequence of discrete outcomes

    Returns:
        (float): the entropy of X
    """
    res = 0
    n = len(X)
    counts = Counter(X).values()
    for count in counts:
        res -= (count / n) * (log(count, 2) - log(n, 2))
    return res


if __name__ == "__main__":
    print(entropy([1, 2, 1, 1, 1, 1]))
    print(entropy([1, 1, 1, 1, 1, 1]))
    print(entropy([1, 1, 1, 2, 2, 2]))
