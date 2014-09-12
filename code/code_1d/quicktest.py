import numpy as np
import numpy.random as rng
import numpy.ma as ma
import pylab as pl
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import matplotlib.mlab as mp 
#import copy, sys
import math
#import optparse
#import scipy.signal
#import scipy.special.basic as sp


def calc_lgamma_vect(vect):
    """Calculate the log gamma of each number in a vector """
    v = np.array(vect) #+ 1e-10
    if np.any(v<0.5): print 'FOUND ELEMENT <0.5 \n %s \n' % v
    for i in range(v.size):       
        v[i] = math.lgamma(v[i])
    return v

def calc_full(n, alphas, sum_alphas, lg_sum_alphas, sum_lg_alphas):
    """ Find logL under DirMult distr. with the alpha parameters supplied."""
    lg_sum_alphas_n = math.lgamma(sum_alphas + np.sum(n))
    sum_lg_alphas_n = np.sum(calc_lgamma_vect(n+alphas))
    s = (lg_sum_alphas - sum_lg_alphas) - (lg_sum_alphas_n - sum_lg_alphas_n)
    return s


def choose(p, arrayShape=(1,)):
    """
    This takes a categorical distribution with parameters p and generates
    i.i.d. samples from it, returning an array of ints in the shape supplied.
    """
    total_num_samples = np.prod(arrayShape)
    a = np.random.multinomial(total_num_samples, p)
    b = np.zeros(total_num_samples, dtype=int)
    upper = np.cumsum(a); lower = upper - a
    for value in range(len(a)):
        b[lower[value]:upper[value]] = value
    np.random.shuffle(b)
    return np.reshape(b,arrayShape)


def expand_boolean_array_one_pixel(m):
    #bigm = np.ndarray(shape = m.shape, dtype=bool)
    bigm = m.copy()
    #bigm[:] = False
    bigm[1:,:] = bigm[1:,:] + m[:-1,:]
    bigm[:-1,:] = bigm[:-1,:] + m[1:,:]
    bigm[:,1:] = bigm[:,1:] + m[:,:-1]
    bigm[:,:-1] = bigm[:,:-1] + m[:,1:]
    bigm[m==True] = False  # doesn't appear to be changing anything?!
    return bigm

if __name__ == "__main__":
    K = 25
    nRows,nCols = 50,60
    # make an image by sampling from a background (BG) distribution
    BG_alphas = 100* np.power(np.arange(K,0,-1.),2.0) # parameters for Dirichlet
    categoricalA = rng.dirichlet(BG_alphas) #sample from Dirichlet
    #print categoricalA , choose(categoricalA) #check...
    z = choose(categoricalA,(nRows,nCols))#samples from categorical
    #print np.histogram(np.ravel(z),bins=-0.5+np.arange(K+1)) #check...
    #
    # make a part of that image by sampling from a source (SRC) distribution
    SRC_alphas = 1.0*np.power(np.arange(K,0,-1.),0.25) # parameters for Dirichlet
    categoricalB = rng.dirichlet(SRC_alphas) #sample from Dirichlet
    print categoricalB , choose(categoricalB) #check...
    SRC_size = [nRows/5,nCols/5]
    tmpz = choose(categoricalB,SRC_size)#samples from categorical
    SRC_posn = [nRows/5,nCols/4]
    bot,top,lft,rgt = SRC_posn[0], SRC_posn[0] + SRC_size[0], SRC_posn[1], SRC_posn[1] + SRC_size[1]
    z[bot:top, lft:rgt] = tmpz
    pl.subplot(221)
    pl.imshow(z, interpolation='nearest',cmap='gray')
    
    # Now! I think I want a map from some key that identifies a putative source to a set of pixels, which is to say a mask. Don't really need a key: just a List of masks? Think so.


    mz = ma.masked_less(z, z.max()) # all z below the max will be masked.
    pl.subplot(222)
    pl.imshow(mz, interpolation='nearest',cmap='gray')
    #list the unmasked locations....
    locations = np.transpose(np.nonzero(~mz.mask))
    print 'locations:'
    print locations

    # expand the mask...
    neighbours = np.logical_not(expand_boolean_array_one_pixel(np.logical_not(mz.mask)))
    pl.subplot(223)
    pl.imshow(neighbours, interpolation='nearest',cmap='gray')
    
    
    pl.savefig('testimg')
    """
    # bogus, but we're setting the background alphas as if there were
    # no sources in the image at the moment....
    alpha_BG = np.histogram(y,bins=BINS)[0] + 0.5
    Cxk = np.zeros((len(BINS)-1,N))
    for i in range(N):
        Cxk[:,i]=np.histogram(y[i],bins=BINS)[0] 


    alpha_SRC = 0.5 * np.ones(alpha_BG.shape)  # 0.5 if the Jeffries prior
    alpha_SRC[0] = 0.5
    for i in range(1,len(alpha_SRC)):
        alpha_SRC[i] = 1.1*alpha_SRC[i-1]
    print 'alpha_SRC: ',alpha_SRC


    #1st two terms for full calculation
    sum_BG = np.sum(alpha_BG)
    lg_sum_BG = math.lgamma(sum_BG)
    sum_lg_BG = np.sum(calc_lgamma_vect(alpha_BG))

    sum_SRC = np.sum(alpha_SRC)
    lg_sum_SRC = math.lgamma(sum_SRC)
    sum_lg_SRC = np.sum(calc_lgamma_vect(alpha_SRC))
      
    nk = 1 # some counts???

    #SCORE                
    SRC_term = calc_full(nk, alpha_SRC, sum_SRC, lg_sum_SRC, sum_lg_SRC)
    BG_term = calc_full(nk, alpha_BG, sum_BG, lg_sum_BG, sum_lg_BG)
    sc = SRC_term - BG_term 
    sc -= np.log(10.0) # this is the effect of our prior on P(source)
                
    score[row,col] = sc









    # where are the brightest pixels in the image?
    mask = np.ndarray(shape = z.shape, dtype=bool)
    mask[:] = False
    mask[z >= z.max()] = True
    """


