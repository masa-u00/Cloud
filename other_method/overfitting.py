#!/usr/bin/python3
# -*- coding: utf-8 -*-
from itertools import product
import sys

import matplotlib.pyplot as plt
import numpy as np

from entropy import entropy
from sc import sc


plt.style.use('ggplot')


def map_randomly(dom_f, img_f):
    f = dict((x, np.random.choice(img_f)) for x in dom_f)
    return f


def generate_cause(src, size):
    if src == "uniform":
        max_X = np.random.randint(2, 10)
        cause = np.array([np.random.randint(1, max_X) for i in range(size)])
    elif src == "multinomial":
        p_nums = [
            np.random.randint(1, 10) for i in range(np.random.randint(3, 4))
        ]
        p_vals = [v / sum(p_nums) for v in p_nums]
        cause = np.random.multinomial(size, p_vals, 1)[0]
        cause = [[i + 1] * f for i, f in enumerate(cause)]
        cause = np.array([j for sublist in cause for j in sublist])
    elif src == "binomial":
        n = np.random.randint(1, 40)
        p = np.random.uniform(0.1, 0.9)
        cause = np.random.binomial(n, p, size)
    elif src == "geometric":
        p = np.random.uniform(0.1, 0.9)
        cause = np.random.geometric(p, size)
    elif src == "hypergeometric":
        ngood = np.random.randint(1, 40)
        nbad = np.random.randint(1, 40)
        nsample = np.random.randint(1, ngood + nbad)
        cause = np.random.hypergeometric(ngood, nbad, nsample, size)
    elif src == "poisson":
        rate = np.random.randint(1, 10)
        cause = np.random.poisson(rate, size)
    elif src == "negativeBinomial":
        n = np.random.randint(1, 40)
        p = np.random.uniform(0.1, 0.9)
        cause = np.random.negative_binomial(n, p, size)
    cause = cause.astype(int)
    return cause


def generate_additive_noise(size):
    t = np.random.randint(1, 4)
    noise = np.array([np.random.randint(-t, t + 1) for i in range(size)])
    noise = noise.astype(int)
    return noise


def are_disjoint(sets):
    disjoint = True
    union = set()
    for s in sets:
        for x in s:
            if x in union:
                disjoint = False
                break
            union.add(x)
    return disjoint


def population():
    # in the population, we know the entropy of noise
    nindividual_population = 100000
    cause = generate_cause("multinomial", nindividual_population)
    support_cause = np.unique(cause)
    support_function_image = range(-2, 2)
    function = map_randomly(support_cause, support_function_image)
    noise = generate_additive_noise(nindividual_population)
    effect = [function[cause[i]] + noise[i]
              for i in range(nindividual_population)]
    return cause, effect, function, noise


def sample(cause_pop, effect_pop, sample_size):
    assert len(cause_pop) == len(effect_pop)
    indices = range(len(cause_pop))
    sample_indices = np.random.choice(indices, sample_size)
    cause_sample = [cause_pop[idx] for idx in sample_indices]
    effect_sample = [effect_pop[idx] for idx in sample_indices]
    return cause_sample, effect_sample


def reliable_entropy(sample):
    return sc(sample) / len(sample)


def function_space(support_cause, support_effect):
    # print(len(support_cause), len(support_effect))
    for item in product(support_effect, repeat=len(support_cause)):
        return list(dict(zip(support_cause, item)))
        # print(list(zip(support_cause, item)))
        # yield dict(zip(support_cause, item))


def minimise_noise(cause_sample, effect_sample, estimator):
    support_cause = set(cause_sample)
    support_effect = set(effect_sample)

    print(support_cause, support_effect)
    min_estimate_noise = sys.float_info.max
    best_function = None

    pair = list(zip(cause_sample, effect_sample))
    for function in func_space:
        noise = [y - function[x] for x, y in pair]
        estimate = estimator(noise)
        if estimate < min_estimate_noise:
            min_estimate_noise = estimate
            best_function = function
            # print(min_estimate_noise)
    # print("\n")
    return best_function


# def noise(cause_sample, effect_sample, function):
#     try:
#         return [y - function[x] for x, y in zip(cause_sample, effect_sample)]
#     except KeyError:
#         print(function)
#         print(set(cause_sample))
#         print(set(effect_sample))
#         raise


if __name__ == "__main__":
    cause_pop, effect_pop, function_pop, noise_pop = population()

    func_space = function_space(set(cause_pop), set(effect_pop))

    entropy_noise_pop = entropy(noise_pop)
    print(entropy_noise_pop)
    # print("\n")

    nsimulation = 100
    sample_sizes = range(10, 1100, 10)
    means_plugin, means_reliable = [], []
    for sample_size in sample_sizes:
        mean_plugin, mean_reliable = 0, 0
        for i in range(nsimulation):
            # print(i, end=" ")
            cause_sample, effect_sample = sample(
                cause_pop, effect_pop, sample_size)

            best_function_plugin = minimise_noise(
                cause_sample, effect_sample, entropy)
            best_function_reliable = minimise_noise(
                cause_sample, effect_sample, reliable_entropy)

            # the domain of estimated function may not be equal to population
            mean_plugin += entropy(noise(cause_pop,
                                         effect_pop, best_function_plugin))
            mean_reliable += entropy(noise(cause_pop,
                                           effect_pop, best_function_reliable))

        means_plugin.append(mean_plugin / nsimulation)
        means_reliable.append(mean_reliable / nsimulation)
        print(sample_size)
        sys.stdout.flush()
    reference = [entropy_noise_pop] * len(sample_sizes)
    plt.plot(sample_sizes, reference)
    plt.plot(sample_sizes, means_plugin)
    plt.plot(sample_sizes, means_reliable)
    plt.legend(["pop", "plugin", "reliable"], loc="lower right")
    plt.show()
