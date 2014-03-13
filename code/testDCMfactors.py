import numpy as np
import numpy.random as rng
import pylab as pl


def calc_logL(n, alphas):
    """ Calculate the log likelihood under DirMult distribution with alphas=avec, given data counts of nvec."""
    lg_sum_alphas = math.lgamma(alphas.sum())
    sum_lg_alphas = np.sum(scipy.special.gammaln(alphas))
    lg_sum_alphas_n = math.lgamma(alphas.sum() + n.sum())
    sum_lg_alphas_n = np.sum(scipy.special.gammaln(n+alphas))
    return lg_sum_alphas - sum_lg_alphas - lg_sum_alphas_n + sum_lg_alphas_n 

if __name__ == "__main__":

    y = np.array([0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0])
    print y
    cutpoint = 4

    # let's start by calculating the (log) prob of y under a DcM with big alphas, say.
    alphas = np.array([10000.0,10000.0])
    dirs = rng.dirichlet(alphas,4)
    print dirs

