import numpy as np
import numpy.random as rng
import numpy.ma as ma
import pylab as pl
import math


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
    bigm = m.copy()
    # First the cells to the Nth, Sth, East and West.
    bigm[1:,:] = bigm[1:,:] + m[:-1,:]
    bigm[:-1,:] = bigm[:-1,:] + m[1:,:]
    bigm[:,1:] = bigm[:,1:] + m[:,:-1]
    bigm[:,:-1] = bigm[:,:-1] + m[:,1:]
    # now the four "diagonal corners"
    bigm[1:,1:] = bigm[1:,1:] + m[:-1,:-1]
    bigm[:-1,:-1] = bigm[:-1,:-1] + m[1:,1:]
    bigm[1:,:-1] = bigm[1:,:-1] + m[:-1,1:]
    bigm[:-1,1:] = bigm[:-1,1:] + m[1:,:-1]
    bigm[m==True] = False  # doesn't appear to be changing anything?!
    return bigm

def calc_Bayes_factor(nk):
        SRC_term = calc_full(nk, model_alpha_SRC, sum_SRC, lg_sum_SRC, sum_lg_SRC)
        BG_term = calc_full(nk, model_alpha_BG, sum_BG, lg_sum_BG, sum_lg_BG)
        sc = SRC_term - BG_term 
        sc -= np.log(1.0) # this is the effect of our prior on P(source)
        return sc


if __name__ == "__main__":
    K = 30
    BINS = -0.5+np.arange(K+1)
    nRows,nCols = 50,60
    # make an image by sampling from a background (BG) distribution
    true_BG_alphas = 100* np.power(np.arange(K,0,-1.),2.0) # parameters for Dirichlet
    categoricalA = rng.dirichlet(true_BG_alphas) #sample from Dirichlet
    #print categoricalA , choose(categoricalA) #check...
    z = choose(categoricalA,(nRows,nCols))#samples from categorical
    #print np.histogram(np.ravel(z),bins=BINS)

    # make a part of that image by sampling from a source (SRC) distribution
    true_SRC_alphas = 1.0*np.power(np.arange(K,0,-1.),0.05) # parameters for Dirichlet
    categoricalB = rng.dirichlet(true_SRC_alphas) #sample from Dirichlet
    print 'source distribution: ',categoricalB , choose(categoricalB) #check...
    SRC_size = [nRows/5,nCols/5]
    tmpz = choose(categoricalB,SRC_size)#samples from categorical
    SRC_posn = [nRows/5,nCols/4]
    bot,top,lft,rgt = SRC_posn[0], SRC_posn[0] + SRC_size[0], SRC_posn[1], SRC_posn[1] + SRC_size[1]
    z[bot:top, lft:rgt] = tmpz
    pl.subplot(221)
    pl.imshow(z, interpolation='nearest',cmap='gray')
    


    # -------------------- now we pretend not to know alphas, or anything! --------------------------
    # bogus, but we're setting the background alphas as if there were
    # no sources in the image at the moment....
    model_alpha_BG = np.histogram(z,bins=BINS)[0] + 0.5
    model_alpha_SRC = 0.5 * np.ones(model_alpha_BG.shape)  # 0.5 if the Jeffries prior
    print 'model_alpha_SRC: ',model_alpha_SRC
    #1st two terms for full calculation
    sum_BG = np.sum(model_alpha_BG)
    lg_sum_BG = math.lgamma(sum_BG)
    sum_lg_BG = np.sum(calc_lgamma_vect(model_alpha_BG))
    sum_SRC = np.sum(model_alpha_SRC)
    lg_sum_SRC = math.lgamma(sum_SRC)
    sum_lg_SRC = np.sum(calc_lgamma_vect(model_alpha_SRC))
      
    threshold = z.max()-3
    mz = ma.masked_less(z, threshold) # all z below this value will be masked.
    locations = np.transpose(np.nonzero(~mz.mask))
    print 'locations of max:', locations

    # pick one location to seed a region
    regions, BFs = [], []
    for seed in locations:
        print seed
        mz = ma.masked_array(z, mask=np.ones(shape=z.shape)) 
        mz.mask[seed[0],seed[1]] = False # unmask just this one spot!

        pl.subplot(222)
        pl.imshow(mz, interpolation='nearest',cmap='gray')

        improvements = True
        while (improvements == True):
            # expand the mask...
            neighbours = np.logical_not(expand_boolean_array_one_pixel(np.logical_not(mz.mask)))

            # We can calculate the current Bayes Factor for the region....
            # First, assess the counts in the region as it stands.
            nk = np.histogram(mz.compressed(),bins=BINS)[0]
            print 'counts: ', nk, nk.sum()
            BF = calc_Bayes_factor(nk)
            print 'bayes_factor is ',BF

            # go through the neighbours in a random order...
            border_sites = np.transpose(np.nonzero(~neighbours))
            rng.shuffle(border_sites)
            N = nk.sum()
            improvements = False
            for site in border_sites:
                k = z[site[0],site[1]]
                BF_increase = math.log((nk[k]+model_alpha_SRC[k])/(N+model_alpha_SRC.sum())) - math.log((nk[k]+model_alpha_BG[k])/(N+model_alpha_BG.sum()))
                #print 'change to Bayes factor should be ', BF_increase
                if BF_increase > 0.0:
                    improvements = True
                    mz.mask[site[0],site[1]] = False #we unmask this pixel!
                    # this is same code as above, so that's not very cool.................
                    nk = np.histogram(mz.compressed(),bins=BINS)[0]
                    print '\t counts: ', nk, nk.sum()
                    BF = calc_Bayes_factor(nk)
                    print '\t bayes_factor is ',BF

        if BF > 0.0:
            regions.append(mz.mask.copy())
            BFs.append(BF)
    
    all_regions_img = np.zeros(shape=z.shape, dtype=float)
    for i in range(len(regions)):
        region_mask = regions[i]
        BF = BFs[i]
        all_regions_img = np.maximum(all_regions_img, BF*(region_mask==False))

    pl.subplot(223)
    pl.imshow(all_regions_img, interpolation='nearest',cmap='hot')
    #pl.colorbar()


    pl.savefig('testimg')


