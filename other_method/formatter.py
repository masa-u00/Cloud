#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module provides common methods for manipulating data"""
from collections import defaultdict


def stratify(X, Y):
    """Stratifies Y based on unique values of X.

    Args:
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes

    Returns:
        (dict): list of Y-values for a X-value
    """
    Y_grps = defaultdict(list)
    for i, x in enumerate(X):
        Y_grps[x].append(Y[i])
    return Y_grps


def to_nested(X):
    """Converts the given sequence to a nested sequence.

    Args:
        X (sequence): sequence of discrete outcomes

    Returns:
        (nested sequence): nested sequence of X
    """
    return [[x] for x in X]
