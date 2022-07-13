from __future__ import division
import sys
import numpy as np
from math import ceil, log, sqrt
from collections import Counter, defaultdict
import itertools

MODEL_CANDIDATES = ["to", "gets", "indep", "confounder"]

def log2(n):
    return log(n or 1, 2)

def C_MN(n: int, K: int):
    """Computes the normalizing term of NML distribution recursively. O(n+K)

    For more detail, please refer to eq (19) (Theorem1) in
    "NML Computation Algorithms for Tree-Structured Multinomial Bayesian Networks"
    https://pubmed.ncbi.nlm.nih.gov/18382603/

    and algorithm 2 in
    "Computing the Multinomial Stochastic Complexity in Sub-Linear Time"
    http://pgm08.cs.aau.dk/Papers/31_Paper.pdf


    Args
    ----------
        n (int): sample size of a dataset
        K (int): K-value multinomal distribution

    Returns
    ----------
        float: (Approximated) Multinomal Normalizing Sum

    """

    total = 1
    b = 1
    d = 16 # 16 digit precision

    #bound = int(ceil(2 + sqrt( -2 * n * np.log(2 * 10**(-d) - 100 ** (-d)))))
    bound = int(ceil(2 + sqrt(2 * n * d * log(10))))  # using equation (38)

    for k in range(1, bound + 1):
        b = (n - k + 1) / n * b
        total += b

    old_sum = 1

    for j in range(3, K + 1):
        new_sum = total + (n * old_sum) / (j - 2)
        old_sum = total
        total = new_sum

    return total

def parametric_complexity(X, Y, model_type: str, X_ndistinct_vals=None, Y_ndistinct_vals=None):
    """Computes the Parametric Complexity of Multinomals.

    Args
    ----------
        model_type (str): ["to", "gets", "indep", "confounder"]
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes
        X_ndistinct_vals (int): number of distinct values of the multinomial r.v X.
        Y_ndistinct_vals (int): number of distinct values of the multinomial r.v Y.

    Returns
    ----------
        float: Parametric Complexity of Multinomals

    """

    assert len(X)==len(Y)
    n = len(X)
    X_ndistinct_vals = X_ndistinct_vals or len(set(X))
    Y_ndistinct_vals = Y_ndistinct_vals or len(set(Y))


    if model_type == "confounder":
        return  log2(C_MN(n=n, K=X_ndistinct_vals * Y_ndistinct_vals))

    else:
        return  log2(C_MN(n=n, K=X_ndistinct_vals)) + log2(C_MN(n=n, K=Y_ndistinct_vals))


# ref: https://github.molgen.mpg.de/EDA/cisc/blob/master/formatter.py
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

def map_to_majority(X, Y):
    """Creates a function that maps x to most frequent y.
    Args:
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes
    Returns:
        (dict): map from Y-values to frequently co-occuring X-values
    """
    f = dict()
    Y_grps = stratify(X, Y)
    for x, Ys in Y_grps.items():
        frequent_y, _ = Counter(Ys).most_common(1)[0]
        f[x] = frequent_y
    return f

def update_regression(C, E, f, max_niterations=10000):
    """Update discrete regression with C as a cause variable and Y as a effect variable
    so that it maximize likelihood
    Args
    -------
        C (sequence): sequence of discrete outcomes
        E (sequence): sequence of discrete outcomes
        f (dict): map from C to Y

    """
    cur_likelihood =  cause_effect_negloglikelihood(C, E, f)
    supp_C = list(set(C))
    supp_E = list(set(E))

    j = 0
    minimized = True
    while j < max_niterations and minimized:
        minimized = False

        for c_to_map in supp_C:
            best_likelihood = sys.float_info.max
            best_e = None

            for cand_e in supp_E:
                if cand_e == f[c_to_map]:
                    continue

                f_ = dict()
                for c_value, e_value in f.items():
                    if c_value == c_to_map:
                        f_[c_value] = cand_e
                    else:
                        f_[c_value] = e_value

                if len(set(f_.values())) == 1:
                    continue
                neglikelihood = cause_effect_negloglikelihood(C, E, f_)

                if neglikelihood < best_likelihood:
                    best_likelihood = neglikelihood
                    best_e = cand_e

            if best_likelihood < cur_likelihood:
                cur_likelihood = best_likelihood
                f[c_to_map] = best_e
                minimized = True
        j += 1

    return f


def cause_effect_negloglikelihood(C, E, func):
    """Compute negative log likelihood for cause & effect pair.
    Model type : C→E

    Args
    -------
        C (sequence): sequence of discrete outcomes (Cause)
        E (sequence): sequence of discrete outcomes (Effect)
        func (dict): map from C-value to E-value

    Returns
    -------
        (float): maximum log likelihood
    """
    mod_C = len(set(C))
    mod_E = len(set(E))
    supp_C = list(set(C))
    supp_E = list(set(E))

    C_freqs = Counter(C)
    E_freqs = Counter(E)
    n = len(C)

    pair_cnt = defaultdict(lambda: defaultdict(int))
    for c, e in zip(C, E):
        pair_cnt[c][e] += 1

    loglikelihood = 0

    for freq in C_freqs.values():
        loglikelihood += freq * (log2(n) - log2(freq))

    for e_E in supp_E:
        freq = 0
        for e in supp_E:
            for c in supp_C:
                if (func[c] + e_E) % mod_E == e:
                    freq += pair_cnt[c][e]
        loglikelihood += freq * (log2(n) - log2(freq))

    return loglikelihood

def neg_log_likelihood(X, Y, model_type):
    """Compute negative maximum log-likelihood of the model given observations z^n.

    Args
    ------
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes
        model_type (str): one of ["to", "gets", "indep", "confounder"]
        f (dict): map from Y-values to frequently co-occuring X-values
    Returns
    -----------
        (float): (negative) maximum log likelihood
    """

    n = len(X)
    loglikelihood = 0

    if model_type == "to":
        f = map_to_majority(X, Y)
        f = update_regression(X, Y, f)
        loglikelihood = cause_effect_negloglikelihood(X, Y, f)

    elif model_type == "gets":
        g = map_to_majority(Y, X)
        g = update_regression(Y, X, g)
        loglikelihood = cause_effect_negloglikelihood(Y, X, g)

    elif model_type == "indep":
        X_freqs = Counter(X)
        Y_freqs = Counter(Y)
        for freq in X_freqs.values():
            loglikelihood += freq * (log2(n) - log2(freq))
        for freq in Y_freqs.values():
            loglikelihood += freq * (log2(n) - log2(freq))

    elif model_type == "confounder":
        pair_cnt = defaultdict(lambda: defaultdict(int))
        for x, y in zip(X, Y):
            pair_cnt[x][y] += 1

        for x in list(set(X)):
            for y in list(set(Y)):
                loglikelihood += pair_cnt[x][y] * (log2(n) - log2(pair_cnt[x][y]))

    return loglikelihood


def sc(X, Y, model_type: str, X_ndistinct_vals=None, Y_ndistinct_vals=None):
    """Computes the stochastic complexity of z^n(two discrete sequences).

    Args
    ------
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes
        model_type (str): ["to", "gets", "indep", "confounder"]
        X_ndistinct_vals (int): number of distinct values of the multinomial r.v X.
        Y_ndistinct_vals (int): number of distinct values of the multinomial r.v Y.

    Returns
    ----------
        float: Stochastic Complexity of a given dataset
    """
    assert len(X)==len(Y)
    X_ndistinct_vals = X_ndistinct_vals or len(set(X))
    Y_ndistinct_vals = Y_ndistinct_vals or len(set(Y))

    data_cost =  neg_log_likelihood(X, Y, model_type)
    model_cost = parametric_complexity(X, Y, model_type, X_ndistinct_vals, Y_ndistinct_vals)

    stochastic_complexity = data_cost + model_cost

    # add function code length
    if model_type == "to":
        stochastic_complexity += log2(Y_ndistinct_vals**(X_ndistinct_vals-1) - 1)
    elif model_type == "gets":
        stochastic_complexity += log2(X_ndistinct_vals**(Y_ndistinct_vals-1) - 1)

    return stochastic_complexity


def ndm(X, Y, ):
    """NML Discrete Model
    """

    results = []

    for model_type in MODEL_CANDIDATES:
        stochastic_complexity = sc(X, Y, model_type)
        results.append((stochastic_complexity, model_type))

    return results


if __name__ == "__main__":
    # unit test
    # usage :
    # $ time python main.py
    import numpy as np
    import argparse
    parser = argparse.ArgumentParser(description="NDM single experiment")
    parser.add_argument("--N", type=int, default=10000, help="number of samples")
    parser.add_argument("--m0", type=int, default=4, help="number of distinct values of the multinomial r.v X")
    parser.add_argument("--m1", type=int, default=5, help="number of distinct values of the multinomial r.v Y")
    args = parser.parse_args()


    # prepare simple dataset
    rand0 = [np.random.random() for _ in range(args.m0)]
    pvals0 = [rand_f / sum(rand0) for rand_f in rand0]
    rand1 = [np.random.random() for _ in range(args.m1)]
    pvals1 = [rand_f / sum(rand1) for rand_f in rand1]
    x0 = np.random.choice(a=range(args.m0), p=pvals0, size=args.N)
    x1 = (x0 + np.random.choice(a=range(args.m1), p=pvals1, size=args.N)) % args.m1

    # unit test for proposed method
    results = ndm(x0, x1)
    results.sort(key=lambda x: x[0])
    print(results)

