from __future__ import division
from math import ceil, log, sqrt
from scipy.stats import binom

def log2(n):
    return log(n or 1, 2)

def C_MN(n: int, K: int):
    """Computes the normalizing term of P_NML recursively. O(n+K)

    For more detail, please refer to eq (19) (Theorem1) in
    "NML Computation Algorithms for Tree-Structured Multinomial Bayesian Networks"
    https://pubmed.ncbi.nlm.nih.gov/18382603/

    Args
    ----------
        n (int): sample size of a dataset
        K (int): K-value multinomal distribution

    Returns
    ----------
        float: Parametric Complexity of Multinomals

    """

    if K == 1:
        C_1 = 1
        return C_1

    elif K == 2:
        # calculate the following equation
        # Σ_(r1+r2=n) n_C_r1 * (r1/n)^r1 * (r2/n)^r2
        C_2 = 0
        for r in range(n+1):
            C_2 += binom.pmf(k=r, n=n, p=r/n) # add nCr (r/n)^r (1 - r/n)^(n-r)
        return C_2

    return C_MN(n=n, K=K-1) + (n/(K-2)) * C_MN(n=n, K=K-2)


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
    X_ndistinct_vals = X_ndistinct_vals or len(np.unique(X))
    Y_ndistinct_vals = Y_ndistinct_vals or len(np.unique(Y))


    if model_type == "confounder":
        return  log2(C_MN(n=n, K=X_ndistinct_vals * Y_ndistinct_vals))

    else:
        return  log2(C_MN(n=n, K=X_ndistinct_vals)) + log2(C_MN(n=n, K=Y_ndistinct_vals))


if __name__ == "__main__":
    # unit test
    # usage :
    # $ time python sc.py
    import numpy as np
    import argparse
    parser = argparse.ArgumentParser(description="BBCI single experiment")
    parser.add_argument("--N", type=int, default=1000, help="number of samples")
    parser.add_argument("--m0", type=int, default=4, help="number of distinct values of the multinomial r.v X")
    parser.add_argument("--m1", type=int, default=5, help="number of distinct values of the multinomial r.v Y")
    args = parser.parse_args([])


    # prepare simple dataset
    x0 = np.random.randint(args.m0, size=args.N)
    x1 = (x0 + np.random.randint(args.m1, size=args.N)) % args.m1


    # unit test for parametric complexity
    pc_cf = parametric_complexity(x0, x1, model_type="confounder")
    print(pc_cf)








