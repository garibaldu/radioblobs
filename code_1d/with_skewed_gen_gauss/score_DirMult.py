import numpy as np
import numpy.random as rng
import pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mp 
import copy, sys
import math
import optparse
import scipy.signal
import scipy.special.basic as sp
import scipy.optimize as sop

source_to_background_ratio = np.log(0.1/0.9)

def sigm(t):
    return 1.0/(1.0+ np.exp(-t))

def skew_gen_gauss(x,mid):

    dev = x - mid
    beta, alpha = (5.-0.5)*rng.random()+0.5, 8.*rng.random()+ 6.
    ggd = beta/(2*alpha*math.gamma(1.0/beta)) * np.exp(-np.power(np.abs(dev)/alpha, beta))
    shape = ggd * sigm(rng.normal()*dev)
    height = (5-0.5)*rng.random()+0.5
    shape = height * shape/shape.max()

    return shape



#TODO: random seed?
def make_dirichlet_bins(data,num_bins,strategy,num_dirs=50,alpha=10.,stretch_factor=None,total_alpha=None,safety_gap=np.inf):

    z = copy.copy(data)
    z.sort()
    top, bottom = z[-1], z[0]

    alphas = [alpha]*num_bins #can only do eqocc and width for now
    dirs = rng.dirichlet(alphas,num_dirs)

    mybins = np.zeros((num_dirs,num_bins+1))
    mybins[:,0]  = bottom
    mybins[:,-1] = top


    if strategy == 'eqocc': #(roughly) equal occupancies
        num_datapts = z.size
        for d in range(dirs.shape[0]):
            props = (np.cumsum(dirs[d])*num_datapts)[:-1]
            for p in range(len(props)):
                mybins[d,p+1] = (z[props[p]] + z[props[p]+1])/2

    elif strategy == 'width': #(roughly) equal width
        datarange = top - bottom
        for d in range(dirs.shape[0]):
            props = np.cumsum(dirs[d])[:-1]
            for p in range(len(props)):
                mybins[d,p+1] = props[p] * datarange

    elif strategy == 'expocc':
        print "strategy expocc not implemented for dirichlet bins yet"
        sys.exit(-1)

    elif strategy == 'dexpocc':
        print "strategy dexpocc not implemented for dirichlet bins yet"
        sys.exit(-1)

    else: 
        sys.exit('Not a valid binning strategy')


    #safety gap
    mybins[:,0]  -= safety_gap
    mybins[:,-1] += safety_gap

    #return bin borders
    return mybins


def make_bin_borders(data,num_bins,strategy='eqocc',safety_gap=np.inf,fname=None,prop=0.5):
    z = copy.copy(data)
    z.sort()
    top, bottom = z[-1], z[0]
    mybins = []

    if strategy == 'eqocc': #Equal occupancies
        step = len(z)/num_bins
        for i in range(0,len(z)-step+1,step):
            mybins.append(z[i])
        mybins.append(z[-1]) # ie. these are really bin BORDERS.

    elif strategy == 'width': #Equal width
        step = (top-bottom)/(num_bins+0.1)
        mybins = [bottom + x*step  for x in range(0, num_bins)]
        mybins.append(z[-1]) # the last one.

    elif strategy == 'expocc':
        # This binning strategy places fewer pixels in each successive
        # bin by a constant multiplicative factor (eg. a half), so it
        # gives exponentially decreasing occupancy. BUT NOTE: with
        # #bins set, AND the factor set, the final bin size CAN'T be.
        i=0
        magic_fraction = prop
        index = 0
        while len(mybins)<num_bins:
            mybins.append(z[index])
            index = min(index+ceil(magic_fraction * (len(z)-index)),  len(z)-1) 
        mybins.append(z[-1]) # ie. these are really bin BORDERS.

    elif strategy == 'dexpocc':
        # As for 'expocc' but the size of the data and the proportion determine
        # num of bins (num bins can't be set by user)
        num = z.size
        last = 0
        mybins.append(z[0])
        while num > 0:
           n = math.ceil(num*prop)
           mybins.append(z[last+n-1])
           last += n
           num -= n

    elif strategy == 'fromfile':
        if fname == None: 
            sys.exit('Please supply a file name')
        else:
            mybins = np.genfromtxt(fname)

    else: 
        sys.exit('Not a valid binning strategy')

    # Now ensure the borders are big enough to catch new data that's out-of-range.
    mybins[-1] += safety_gap
    mybins[0]  -= safety_gap


    return mybins

def get_BG(fname):
    """Get background alpha vector from LDA output"""
    CWT = np.delete(np.delete(np.genfromtxt(fname,comments='#'),0,1),0,0)
    #"biggest" topic is background (return this as background alpha vector)
    t0 = CWT[:,0]
    t1 = CWT[:,1]
    if np.sum(t0) > np.sum(t1):
        return t0
    else:
        return t1

def make_alphaBG(BINS,N,Z,dirichlet):

    if dirichlet:
        alpha_BGs = np.zeros((BINS.shape[0],BINS.shape[1]-1))
        K = BINS.shape[1]-1
        Cxk = np.zeros((N,K))
        for b in range(BINS.shape[0]):
            alpha_BGs[b] = np.histogram(np.ravel(Z),bins=BINS[b])[0]
            for i in range(K-1):
                Cxk[:,i]+=np.asarray((Z>=BINS[b,i])&(Z<BINS[b,i+1]),dtype=int)
            Cxk[:,K-1]+=np.asarray((Z>=BINS[b,K-1])&(Z<=BINS[b,K]),dtype=int)
        alpha_BG = np.mean(alpha_BGs,axis=0) + 1.0
        Cxk /= float(BINS.shape[0])


    else:
        alpha_BG = np.histogram(np.ravel(Z),bins=BINS)[0] + 1.0
        K = len(BINS)-1
        Cxk = np.zeros((N,K))
        for i in range(K-1):
            Cxk[:,i]=np.asarray((Z>=BINS[i])&(Z<BINS[i+1]),dtype=int)
        Cxk[:,K-1]=np.asarray((Z>=BINS[K-1])&(Z<=BINS[K]),dtype=int)

    Cxk = Cxk.T
    return Cxk, alpha_BG

############################ SCORE METHODS ############################

def calc_params(md, sigma):

    Cxk_slice = Cxk

    xb = np.arange(0,N,dtype='float')
    wgts = np.exp(-(np.power((xb-md),2.)/(2.*np.power(sigma,2.))))

    nk = np.sum(wgts*Cxk_slice,axis=1)

    return Cxk_slice, xb, wgts, nk


def calc_full(n, alphas):
    """ Calculate the log likelihood under DirMult distribution with alphas=avec, given data counts of nvec."""
    lg_sum_alphas = math.lgamma(alphas.sum())
    sum_lg_alphas = np.sum(scipy.special.gammaln(alphas))
    lg_sum_alphas_n = math.lgamma(alphas.sum() + n.sum())
    sum_lg_alphas_n = np.sum(scipy.special.gammaln(n+alphas))
    return lg_sum_alphas - sum_lg_alphas - lg_sum_alphas_n + sum_lg_alphas_n 



def score_wrapper(theta, args):
    """ Calculate and return the score """ 

    md,sigma = theta
    alpha_SRC,alpha_BG = args

    Cxk_slice, xb, wgts, nk = calc_params(md, sigma)

    SRC_term = calc_full(nk, alpha_SRC)
    BG_term = calc_full(nk, alpha_BG)
    s1 = SRC_term - BG_term

    nk_n = (alpha_BG/alpha_BG.sum()) * np.sum(nk)
    SRC_term_n = calc_full(nk_n, alpha_SRC)
    BG_term_n = calc_full(nk_n, alpha_BG)
    s2 = SRC_term_n - BG_term_n

    score = s1 - s2

    return -score


############################ GRADIENT METHODS ############################

def calc_gradients(x,sigma,m,wx):
    """ Calculate gradients for m and sigma for a given m, sigma, and window[xposl:xposr]
        Returns two x-length vectors."""

    grad_m = (wx*(x-m))/(np.power(sigma,2.))
    grad_sigma = (wx*(np.power((m-x),2.)))/(np.power(sigma,3.))

    return grad_m, grad_sigma

def calc_grad_weight(nks, alphaS, alphaB, N, AB, AS):
    """ Calculate the weights for each bin k. Returns k-length vector."""
    K = nks.size
    w = sp.psi(nks + alphaS) - sp.psi(nks+alphaB) + sp.psi(N+AB) - sp.psi(N+AS)

    return w

def calc_fullgrad(wgt,data,gradient):
    """ Calculate full gradient: wgts * (data * grad) """   
    full = np.dot(wgt, (np.sum(data*gradient,axis=1) ))

    return full


def gradient_wrapper(theta, args):
    """ Calculate and return the gradient """

    md,sigma = theta
    alpha_SRC,alpha_BG = args

    Cxk_slice, xb, wgts, nk = calc_params(md, sigma)

    grad_m,grad_sigma = calc_gradients(xb,sigma,md,wgts)
    w = calc_grad_weight(nk,alpha_SRC,alpha_BG,np.sum(nk),alpha_BG.sum(),alpha_SRC.sum())
    gm = calc_fullgrad(w,Cxk_slice,grad_m)
    gs = calc_fullgrad(w,Cxk_slice,grad_sigma)

    nk_n = (alpha_BG/alpha_BG.sum()) * np.sum(nk)
    wn = calc_grad_weight(nk_n,alpha_SRC,alpha_BG,np.sum(nk_n),alpha_BG.sum(),alpha_SRC.sum())
    gmn = calc_fullgrad(wn,Cxk_slice,grad_m)
    gsn = calc_fullgrad(wn,Cxk_slice,grad_sigma)

    return [-(gm-gmn),-(gs-gsn)]



if __name__ == "__main__":

    parser = optparse.OptionParser(usage="usage %prog [options]")

    parser.add_option("-n","--numbins",type = "int",dest = "K",default=0,
                      help="number of bins (ignored if strategy is dexpocc or fromfile)")
    parser.add_option("-b","--bins_fname",dest = "bfname",
                      help="bin borders filename")   
    parser.add_option("-s","--binning_strategy",dest = "strategy",
                      help="eqocc, width, expocc, dexpocc or fromfile. " 
                           "MANDATORY OPTION.")
    parser.add_option("-p","--prop",type="float",dest="prop",default=0.5,
                      help="proportion to decrease bin occupancy by (for use "
                           "with dexpocc; else ignored. DEFAULT VALUE = 0.5)")
    parser.add_option("-d","--datafile",dest = "infile",
                      help="a list of numbers: 1D data to be read in (can't be "
                           "used with --rngseed)")
    parser.add_option("-r","--rngseed",type = "int",dest = "seed",
                      help="an int to make random data up (can't be used with "
                           "--datafile)")
    parser.add_option("-q","--hard",action="store_true",dest="hard",default=False,
                      help="make hard/rectangular windows (default = soft/squared"
                           " exponential)")
    parser.add_option("-t","--dirichlet",action="store_true",dest="dirichlet",default=False,
                      help="make dirichlet bin borders (incompatible with \"from file\" binning stratgegy)")
    parser.add_option("-o","--nohisto",action="store_true",dest="nohisto",default=False,
                      help="no histo in fig")
    parser.add_option("-C","--CWT_fname",dest="CWT",
                      help="give CWT filename if background alphas from LDA "
                           "file to be used (can't be used with --local or --seed)\n")

    opts, args = parser.parse_args()

    EXIT = False

    if opts.strategy is None:
        print "ERROR: you must supply a binning strategy\n"
        EXIT = True

    if opts.infile and opts.seed:
        print "ERROR: supply EITHER a datafile OR a random seed to make up data\n"
        EXIT = True

    if opts.seed and opts.CWT:
        print "ERROR: background alphas from CWT can't be used with randomly generated data\n"
        EXIT = True

    if opts.dirichlet and opts.strategy=="fromfile":
        print "ERROR: dirichlet bin borders are incompatible with using bin borders from file\n"
        EXIT = True

    if EXIT: 
        parser.print_help()
        sys.exit(-1)

    strategy = opts.strategy
    outfile = 'DirModel_%s' % strategy
    K = opts.K

    if opts.seed:
        seed = opts.seed
        # make an "image"
        rng.seed(seed)  # seed the random number generator here

        N = 500 #number of pixels in a fake test image
        noise_size=1.0
        x = np.arange(N)
        # make up the 'shapes' of the sources
        mid1, mid2, mid3 = (N-20)*rng.random()+10,(N-20)*rng.random()+10,(N-20)*rng.random()+10
        print 'Random sources placed at ',mid1, mid2, mid3
        spread1 = 8.*rng.random()+ 6.  # length scale
        shape1 = (4.5*rng.rand()+0.5)*np.exp(-0.5*np.power((x-mid1)*1.0/spread1,2.0)) 
        shape2 = skew_gen_gauss(x,mid2)
        shape3 = skew_gen_gauss(x,mid3)

        # noise character of sources
        variance = np.abs(noise_size*(1.0 - shape1 + shape2)) # source 3 has no variance effect
        #variance = variance + x/float(len(x)) # to mimic steady change over large scales
        noise = rng.normal(0,variance,x.shape)
        # mean_intensity character of sources
        mean = shape1 + shape2 + shape3
        y = mean + noise
  
        outfile += '_r%d_m%d-%d-%d' % (seed, int(mid1), int(mid2), int(mid3))


    else:    # it's not a digit, so it's a filename. File should be just list of numbers.
        infile = opts.infile
        y = np.genfromtxt(infile)
        x = np.arange(len(y))
        N = len(y)

        outfile += '_%s' % infile


    #make bins (here, from the naked image)
    if opts.dirichlet:
        outfile += '_dirichletborders'
        BINS = make_dirichlet_bins(y,K,strategy)
        if K == 0:
            K = BINS.shape[1] - 1
        print 'Note: an example overall histogram: (using the first of the dirichlet histograms)'
        print np.histogram(y,bins=BINS[0])[0]
    else:
        BINS = make_bin_borders(y,K,strategy,safety_gap=np.inf,fname=opts.bfname,prop=opts.prop)
        if K == 0:
            K =  len(BINS) - 1
        print 'Note: this makes the overall histogram this: (reality-check the final one especially)'
        print np.histogram(y,bins=BINS)[0]

    outfile += '_K%d' % K

    #get background alphas from LDA output, if specified
    if opts.CWT:
        alpha_BG = get_BG(opts.CWT)
        outfile += '_LDA'
    else:
        # bogus, but we're setting the background alphas as if there were
        # no sources in the image at the moment....
        Cxk,alpha_BG = make_alphaBG(BINS,N,y,opts.dirichlet)


    alpha_SRC = 1.0 * np.ones(alpha_BG.shape)  

    max_spread = N
    max_wd = max_spread/2

    outfile += '_fullscore'
     

    if opts.hard:
        outfile += '_hardborders'

    outfile += '_optima'

    #do gradient ascent
    num_top = 3
    num_iters = 50
    messages = np.zeros(9)
    top_scores=np.zeros((num_top,5))

    # m bounds: 0,max_spread; sigma bounds: 0,max_wd
    Bounds = [(0,max_spread),(0,max_wd)]

    print "gradient descent ... "

    #np.seterr(divide='raise')

    for i in range(num_top):

        optima = np.zeros((num_iters,5))
        for j in range(num_iters):
            print '------------------------------------------\niter %s.%s\n' % (i,j)
            md = int(rng.rand()*max_spread)
            sigma = int(rng.rand()*(max_wd/10.))

            print '\nmd: %s, sigma: %s\n' % (md, sigma)
    
            theta = [md,sigma]

            args = [alpha_SRC, alpha_BG]

            #sltn, its, rc = sop.fmin_tnc(score_wrapper, theta, gradient_wrapper, [args], bounds=Bounds, maxfun=1000, fmin=-1e10)
            sltn, its, rc = sop.fmin_tnc(score_wrapper, theta, args=[args], approx_grad=True, bounds=Bounds, fmin=-1e10,accuracy=1e-16)
            sc = score_wrapper(sltn, args)
            optima[j,:2] = sltn
            optima[j,2] = -sc
            optima[j,3:] = gradient_wrapper(sltn,args)
            messages[rc] += 1


        top_opt = scipy.delete(optima, np.where(np.isnan(optima)), 0)
        top_opt = top_opt[np.argsort(top_opt[:,2])][-1]
        top_scores[i] = top_opt
        #remove best source
        top_md=top_opt[0]
        top_sig=top_opt[1]
        y[top_md-top_sig:top_md+top_sig+1]=np.nan
        Cxk,alpha_BGs = make_alphaBG(BINS,N,y,opts.dirichlet)

    print '%s local minimum' % messages[0]
    print '%s fconverged' % messages[1]
    print '%s xconverged' % messages[2]
    print '%s max functions reached' % messages[3]
    print '%s linear search failed' % messages[4]
    print '%s constant' % messages[5]
    print '%s no progress' % messages[6]
    print '%s user aborted' % messages[7]
    print '%s infeasible' % messages[8]

    for i in range(top_scores.shape[0]):
        print top_scores[i] 
 
    np.savetxt(outfile,top_scores)

    plt.clf()
    #data
    plt.plot(y,'k.')
    plt.plot(shape1,'b-') 
    plt.plot(shape2,'b-')
    plt.plot(shape3,'b-')
    #optima
    found1 = np.exp(-0.5*np.power((x-top_scores[0,0])*1.0/top_scores[0,1],2.0)) 
    plt.plot(found1,'r-')
    found2 = np.exp(-0.5*np.power((x-top_scores[1,0])*1.0/top_scores[1,1],2.0)) 
    plt.plot(found2,'r-')
    found3 = np.exp(-0.5*np.power((x-top_scores[2,0])*1.0/top_scores[2,1],2.0)) 
    plt.plot(found3,'r-')
    plt.savefig(outfile)

    
    #print shapes
    outfile += '_GT'
    with file(outfile, 'w') as out:
        out.write('# %s %s \n' % (mid1, spread1))
        np.savetxt(out, shape1) 
        out.write('# %s \n' % (mid2))
        np.savetxt(out, shape2) 
        out.write('# %s \n' % (mid3))
        np.savetxt(out, shape3) 


