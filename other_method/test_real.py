#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module assess the performance of various discrete causal inference
methods on real-world discrete cause-effect pairs.
"""
from __future__ import division
import enum
import os

import numpy as np
import pandas as pd

from anm import anm
from cisc import cisc
from dc import dc
from entropic import entropic
from formatter import to_nested
from measures import ChiSquaredTest, Entropy, StochasticComplexity


class ScoreType(enum.Enum):
    PVAL = 1,
    INFO = 2


def pprint(method_name, score, score_type, llabel, rlabel):
    level = 0.05
    arrow = "~"
    if score_type == ScoreType.INFO:
        if score[0] < score[1]:
            arrow = "⇒"
        elif score[0] > score[1]:
            arrow = "⇐"
        conf = abs(score[0] - score[1])
        out_str = "%s:: %s %s %s\t Δ=%.2f" % (method_name,
                                              llabel, arrow, rlabel, conf)
    elif score_type == ScoreType.PVAL:
        if score[0] > level and score[1] < level:
            arrow = "⇒"
        elif score[0] < level and score[1] > level:
            arrow = "⇐"
        out_str = "%s:: %s %s %s" % (method_name, llabel, arrow, rlabel)
    print(out_str)


def test_car():
    print("evaluating on car dataset")
    car_dir = os.path.join(os.path.dirname(__file__), "data", "car")
    car_dat_path = os.path.join(car_dir, "car.dat")

    X_labels = ["buying price", "maintenance",
                "#dorrs", "capacity", "luggage boot", "safety"]
    Y_label = "car acceptibility"
    data = np.loadtxt(car_dat_path)
    nattr = data.shape[1]
    Y = data[:, nattr - 1]
    for i in range(nattr - 2, nattr - 1):
        X = data[:, i]
        dc_score = dc(to_nested(X), to_nested(Y))
        ent_score = entropic(pd.DataFrame(np.column_stack((X, Y))))
        dr_score = anm(X, Y, ChiSquaredTest)
        cisc_score = cisc(X, Y)
        acid_score = anm(X, Y, Entropy)
        crisp_score = anm(X, Y, StochasticComplexity, enc_func=True)

        pprint("   dc", dc_score, ScoreType.INFO, X_labels[i], Y_label)
        pprint("  ent", ent_score, ScoreType.INFO, X_labels[i], Y_label)
        pprint("   dr", dr_score, ScoreType.PVAL, X_labels[i], Y_label)
        pprint(" cisc", cisc_score, ScoreType.INFO, X_labels[i], Y_label)
        pprint(" acid", acid_score, ScoreType.INFO, X_labels[i], Y_label)
        pprint("crisp", crisp_score, ScoreType.INFO, X_labels[i], Y_label)


def test_abalone():
    # We do not discretise the variables just as what DR paper does
    print("evaluating on abalone dataset")
    abalone_dir = os.path.join(os.path.dirname(__file__), "data", "abalone")
    abalone_dat_path = os.path.join(abalone_dir, "abalone.dat")
    data = np.loadtxt(abalone_dat_path)
    ncols = data.shape[1]
    X = data[:, 0]

    colnames = ["sex", "length", "diameter", "height"]
    for i in range(1, ncols):
        Y = data[:, i]
        dc_score = dc(to_nested(X), to_nested(Y))
        ent_score = entropic(pd.DataFrame(np.column_stack((X, Y))))
        dr_score = anm(X, Y, ChiSquaredTest)
        cisc_score = cisc(X, Y)
        acid_score = anm(X, Y, Entropy)
        crisp_score = anm(X, Y, StochasticComplexity, enc_func=True)

        pprint("   dc", dc_score, ScoreType.INFO, colnames[0], colnames[i])
        pprint("  ent", ent_score, ScoreType.INFO, colnames[0], colnames[i])
        pprint("   dr", dr_score, ScoreType.PVAL, colnames[0], colnames[i])
        pprint(" cisc", cisc_score, ScoreType.INFO, colnames[0], colnames[i])
        pprint(" acid", acid_score, ScoreType.INFO, colnames[0], colnames[i])
        pprint("crisp", crisp_score, ScoreType.INFO, colnames[0], colnames[i])


def test_nlschools():
    print("evaluating on nlschools dataset")
    abalone_dir = os.path.join(os.path.dirname(__file__), "data", "nlschools")
    abalone_dat_path = os.path.join(abalone_dir, "nlschools.dat")
    data = np.loadtxt(abalone_dat_path)
    X = data[:, 0]
    Y = data[:, 1]

    dc_score = dc(to_nested(X), to_nested(Y))
    ent_score = entropic(pd.DataFrame(np.column_stack((X, Y))))
    dr_score = anm(X, Y, ChiSquaredTest)
    cisc_score = cisc(X, Y)
    acid_score = anm(X, Y, Entropy)
    crisp_score = anm(X, Y, StochasticComplexity, enc_func=True)

    pprint("   dc", dc_score, ScoreType.INFO, "score", "status")
    pprint("  ent", ent_score, ScoreType.INFO, "score", "status")
    pprint("   dr", dr_score, ScoreType.PVAL, "score", "status")
    pprint(" cisc", cisc_score, ScoreType.INFO, "score", "status")
    pprint(" acid", acid_score, ScoreType.INFO, "score", "status")
    pprint("crisp", crisp_score, ScoreType.INFO, "score", "status")


def test_horse_colic():
    print("evaluating on horse colic dataset")
    horse_dir = os.path.join(os.path.dirname(__file__), "data", "horse")

    def get_data(fname):
        horse_dat_path = os.path.join(horse_dir, fname)
        data = np.genfromtxt(horse_dat_path)
        abdomen = data[:, 17]
        surgical = data[:, 23]
        missing_indices = np.argwhere(np.isnan(abdomen))
        abdomen = np.delete(abdomen, missing_indices)
        surgical = np.delete(surgical, missing_indices)
        return abdomen, surgical

    abdomen_train, surgical_train = get_data("horse_train.dat")
    abdomen_test, surgical_test = get_data("horse_test.dat")
    abdomen = np.concatenate((abdomen_train, abdomen_test))
    surgical = np.concatenate((surgical_train, surgical_test))
    assert len(abdomen) == len(surgical)

    dc_score = dc(to_nested(abdomen), to_nested(surgical))
    ent_score = entropic(pd.DataFrame(np.column_stack((abdomen, surgical))))
    dr_score = anm(abdomen, surgical, ChiSquaredTest)
    cisc_score = cisc(abdomen, surgical)
    acid_score = anm(abdomen, surgical, Entropy)
    crisp_score = anm(abdomen, surgical, StochasticComplexity, enc_func=True)

    pprint("   dc", dc_score, ScoreType.INFO, "abdomen", "surgical")
    pprint("  ent", ent_score, ScoreType.INFO, "abdomen", "surgical")
    pprint("   dr", dr_score, ScoreType.PVAL, "abdomen", "surgical")
    pprint(" cisc", cisc_score, ScoreType.INFO, "abdomen", "surgical")
    pprint(" acid", acid_score, ScoreType.INFO, "abdomen", "surgical")
    pprint("crisp", crisp_score, ScoreType.INFO, "abdomen", "surgical")


if __name__ == "__main__":
    test_horse_colic()
    test_car()
    test_abalone()
    test_nlschools()
