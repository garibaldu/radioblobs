import numpy as np
import numpy.random as rng
import pylab as pl
import math, scipy 
import scipy.special

def calc_logL(n, alphas):
    """ Calculate the log likelihood under DirMult distribution with alphas=avec, given data counts of nvec."""
    lg_sum_alphas = math.lgamma(alphas.sum())
    sum_lg_alphas = np.sum(scipy.special.gammaln(alphas))
    lg_sum_alphas_n = math.lgamma(alphas.sum() + n.sum())
    sum_lg_alphas_n = np.sum(scipy.special.gammaln(n+alphas))
    return lg_sum_alphas - sum_lg_alphas - lg_sum_alphas_n + sum_lg_alphas_n 

import numpy as np

def unique_count(a):  # needs python 1.8
    unique, inverse = np.unique(a, return_inverse=True)
    count = np.zeros(len(unique), np.int)
    np.add.at(count, inverse, 1)
    return np.vstack(( unique, count)).T

def do_example(yvals, f):
    alphas = f*np.ones((2),dtype=float)
    #print 'samples from Dirichlet: \n', rng.dirichlet(alphas,4)
    counts, bns = np.histogram(yvals, bins=y.max()+1)
    #print counts
    return calc_logL(counts, alphas)


########################################
if __name__ == "__main__":


    #print unique_count(np.random.randint(-10,10,100))

    print '-----------------------------'

    y = np.array([0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0])
    print y
    cutpoint = 4
    y1, y2 = y[:cutpoint], y[cutpoint:]

    for f in [1,10,100,1000,10000]:
        logL1 = do_example(y1, f)
        logL2 = do_example(y2, f)
        logL  = do_example(y, f)
        print 'mis-match for alphas %d: '%(f), (logL1 + logL2) - logL


