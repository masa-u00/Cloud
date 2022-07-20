#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""All the utility functions used everywhere in this project are here.
"""
import sys

import matplotlib as mp
import matplotlib.pyplot as plt


def _configure_matplotlib():
    plt.style.use('ggplot')
    font_size = 13
    fig = mp.pyplot.gcf()
    fig.set_size_inches(18, 12)


_configure_matplotlib()


def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben


def plot_multiline(Ys, X, labels, xlabel, ylabel, title, fig_name=None):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for i, Y in enumerate(Ys):
        plt.plot(X, Y, label=labels[i], marker="o", linewidth=2.0,
                 dash_joinstyle="bevel", solid_capstyle="round", markeredgecolor="none", linestyle="-")
        plt.legend(loc="upper left")
    plt.ylim(0.0, 1.001)
    if fig_name:
        plt.savefig("results/%s" % fig_name)
    else:
        plt.show()
    plt.cla()


def reverse_argsort(X):
    indices = range(len(X))
    indices.sort(key=X.__getitem__, reverse=True)
    return indices


def dc_compat(X):
    return [[x] for x in X]
