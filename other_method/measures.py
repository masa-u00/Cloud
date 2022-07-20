#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module abstracts information measures."""
import abc
from enum import Enum

import numpy as np
from scipy.stats import chi2_contingency

from entropy import entropy
from sc import sc


class DMType(Enum):
    NHST = 1    # Null Hypothesis Significance Testing
    INFO = 2    # Information-theoretic


class DependenceMeasure(abc.ABC):

    @property
    @abc.abstractmethod
    def type(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def measure(seq1, seq2=None):
        pass


class Entropy(DependenceMeasure):
    type = DMType.INFO

    def measure(seq1, seq2=None):
        return entropy(seq1)


class StochasticComplexity(DependenceMeasure):
    type = DMType.INFO

    def measure(seq1, seq2=None):
        return sc(seq1)


class ChiSquaredTest(DependenceMeasure):
    type = DMType.NHST

    @staticmethod
    def contingency_table(seq1, seq2):
        dom_seq1 = list(set(seq1))
        dom_seq2 = list(set(seq2))

        ndom_seq1 = len(dom_seq1)
        ndom_seq2 = len(dom_seq2)

        indices1 = dict(zip(dom_seq1, range(ndom_seq1)))
        indices2 = dict(zip(dom_seq2, range(ndom_seq2)))

        table = np.zeros((ndom_seq1, ndom_seq2))

        for k, v1 in enumerate(seq1):
            v2 = seq2[k]
            i, j = indices1[v1], indices2[v2]
            table[i, j] += 1

        return table

    @staticmethod
    def nhst(seq1, seq2):
        assert len(seq1) == len(seq2), "samples are not of the same size"

        table = ChiSquaredTest.contingency_table(seq1, seq2)
        chi2, p_value, _, _ = chi2_contingency(table, correction=True)
        return chi2, p_value

    @staticmethod
    def measure(seq1, seq2=None):
        chi2, p_value = ChiSquaredTest.nhst(seq1, seq2)

        # we want to minimise the dependence between seq1 and seq2 in ANM
        # that is, maximise the independence between seq1 and seq2 in ANM
        # H0: seq1 and seq2 are independent
        # H0 becomes true if p-value is greater than a threshold
        # thus we want to maximise p-value, or minimise the negative of p-value
        p_value *= -1

        # as chi2 gets smaller, p-value increases (check the chi2 plot in wiki)
        # if the p-value is too small, we reject H0 anyway
        # in such a case, we want to minimise chi2
        if p_value < 10 ** -16:
            p_value = chi2
        return p_value


if __name__ == "__main__":
    print(ChiSquaredTest.measure(np.random.choice(
        [1, 2, 3], 10), np.random.choice([1, 2], 10)))
    print(Entropy.measure(np.random.choice([1, 2, 3], 10)))
    print(StochasticComplexity.measure(np.random.choice([1, 2, 3], 10)))
