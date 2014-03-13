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

    else: 
        sys.exit('Not a valid binning strategy')

    # Now ensure the borders are big enough to catch new data that's out-of-range.
    mybins[-1] += safety_gap
    mybins[0]  -= safety_gap


    return mybins


def calc_logP(n, alphas):
    """ Calculate the log likelihood under DirMult distribution with alphas=avec, given data counts of nvec. 
        Uses approximation of log gamma"""
    a = alphas + 0.0001 # just in case of zeros, which log objects to working with.
    #print a,n
    Ntot = np.sum(n)
    Atot = np.sum(a)
    s = Atot*np.log(Atot)  - (Ntot+Atot)*np.log(Ntot+Atot) + Ntot
    s = s + np.sum((n+a)*np.log(n+a) - n - a*np.log(a))
    return  s


def calc_lgamma_vect(vect):
    """Calculate the log gamma of each number in a vector """
    v = vect + 0.0001 #in case of zeros, which log gamma doesn't like
    for i in range(v.size):
        v[i] = math.lgamma(v[i])
    return v


def calc_full(n, alphas, sum_alphas, lg_sum_alphas, sum_lg_alphas):
    """ Calculate the log likelihood under DirMult distribution with alphas=avec, given data counts of nvec."""
    lg_sum_alphas_n = math.lgamma(sum_alphas + np.sum(n))
    sum_lg_alphas_n = np.sum(calc_lgamma_vect(n+alphas))
    s = lg_sum_alphas - sum_lg_alphas - lg_sum_alphas_n + sum_lg_alphas_n
    return s

def calc_gradients(x,sigma,m):
    """ Calculate gradients for m and sigma for a given m, sigma, and window[xposl:xposr]
        Returns two x-length vectors."""
    wx = np.exp(-(np.power(x-m,2.)/2.*np.power(sigma,2.)))

    grad_m = wx*(x-m)/np.power(sigma,2.)
    grad_sigma = wx*np.power((x-m),2.)/np.power(sigma,3.)
    
    return grad_m, grad_sigma

def calc_grad_weight(nks, alphaS, alphaB, N, AB, AS):
    """ Calculate the weights for each bin k. Returns k-length vector."""
    w = sp.psi(nks + alphaS) - sp.psi(nks+alphaB) + sp.psi(N+AB) - sp.psi(N+AS)
    return w

def calc_fullgrad(wgt,data,gradient):

    full = np.dot(wgt,np.dot(data,gradient))
    return full

if __name__ == "__main__":

    parser = optparse.OptionParser(usage="usage %prog [options]")

    parser.add_option("-n","--numbins",type = "int",dest = "K",default=0,
                      help="number of bins (ignored if strategy is dexpocc or fromfile)")
    parser.add_option("-s","--binning_strategy",dest = "strategy",
                      help="eqocc, width or fromfile. " 
                           "MANDATORY OPTION.")
    parser.add_option("-d","--datafile",dest = "infile",
                      help="a list of numbers: 1D data to be read in (can't be "
                           "used with --rngseed)")
    parser.add_option("-r","--rngseed",type = "int",dest = "seed",
                      help="an int to make random data up (can't be used with "
                           "--datafile)")
    parser.add_option("-t","--dirichlet",action="store_true",dest="dirichlet",default=False,
                      help="make dirichlet bin borders (incompatible with \"from file\" binning stratgegy)")
    parser.add_option("-o","--nohisto",action="store_true",dest="nohisto",default=False,
                      help="no histo in fig")

    opts, args = parser.parse_args()

    EXIT = False

    if opts.strategy is None:
        print "ERROR: you must supply a binning strategy\n"
        EXIT = True

    if opts.infile and opts.seed:
        print "ERROR: supply EITHER a datafile OR a random seed to make up data\n"
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
        mid1, mid2, mid3 = rng.random() * N,rng.random() * N,rng.random() * N
        print 'Random sources placed at ',mid1, mid2, mid3
        spread1 = int(N/80 + rng.random()*N/50)  # length scale
        spread2 = int(2+2*rng.random())  # length scale
        spread3 = int(2+2*rng.random())  # length scale
        shape1 = 0.8*np.exp(-0.5*np.power((x-mid1)*1.0/spread1,2.0))
        shape2 = 5.0*np.exp(-0.5*np.power((x-mid2)*1.0/spread2,2.0))
        shape3 = 3.0*np.exp(-0.5*np.power((x-mid3)*1.0/spread3,2.0))
        # noise character of sources
        variance = noise_size*(1.0 - shape1 + shape2) # source 3 has no variance effect
        #variance = variance + x/float(len(x)) # to mimic steady change over large scales
        noise = rng.normal(0,variance,x.shape)
        # mean_intensity character of sources
        mean = shape1 + shape2 + shape3
        y = mean + noise

        outfile += '_%d' % seed

        #shapex left and right is +/- 1 sigma from the mean
        #gives three true [leftx,rightx] for three shapes
        left1=round(mid1)-spread1; right1=round(mid1)+spread1+1
        left2=round(mid2)-spread2; right2=round(mid2)+spread2+1
        left3=round(mid3)-spread3; right3=round(mid3)+spread3+1


        true_sources = [(left1,right1),(left2,right2),(left3,right3)]
        true_sources.sort()

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

    # bogus, but we're setting the background alphas as if there were
    # no sources in the image at the moment....
    if opts.dirichlet:
        alpha_BGs = np.zeros((BINS.shape[0],BINS.shape[1]-1))
        Cxk = np.zeros((BINS.shape[1]-1,N))
        for b in range(BINS.shape[0]):
            alpha_BGs[b] = np.histogram(y,bins=BINS[b])[0]
            for i in range(N):
                Cxk[:,i] += np.histogram(y[i],bins=BINS[b])[0]
        alpha_BG = np.mean(alpha_BGs,axis=0) + 0.5
        Cxk /= float(BINS.shape[0])
    else:
        alpha_BG = np.histogram(y,bins=BINS)[0] + 0.5
        Cxk = np.zeros((len(BINS)-1,N))
        for i in range(N):
            Cxk[:,i]=np.histogram(y[i],bins=BINS)[0] 

    alpha_SRC = 0.5 * np.ones(alpha_BG.shape)  # 0.5 if the Jeffries prior

    max_spread = N
    score = np.zeros((max_spread,N))
    max_wd = max_spread/2
    gradients = np.zeros((max_wd,max_spread,2))


    outfile += '_fullscore'
    #1st two terms for full calculation
    sum_BG = np.sum(alpha_BG)
    lg_sum_BG = math.lgamma(sum_BG)
    sum_lg_BG = np.sum(calc_lgamma_vect(alpha_BG))

    sum_SRC = np.sum(alpha_SRC)
    lg_sum_SRC = math.lgamma(sum_SRC)
    sum_lg_SRC = np.sum(calc_lgamma_vect(alpha_SRC))
      

    score1d = np.zeros((N))
    sig_wd = 3.
    print max_wd
    for half_wd in range(1,max_wd,1): # wd is width of the window
        for col in range(max_spread):
            # evaluate the score of a model that has middle=row, spread=col.
            md = col
            row = half_wd

            lo = max(0, md - sig_wd*half_wd)
            hi = min(N-1, md + sig_wd*half_wd)
            
            if ((hi - lo)>1) and (md >= lo) and (md <= hi):
                # otherwise it's a fairly silly model!
                bound = y[lo:hi+1]
                Cxk_slice = Cxk[:,lo:hi+1]
                win_size = half_wd*(2*sig_wd) + 1

                wgts = scipy.signal.gaussian(win_size,half_wd)

                l = math.floor((len(wgts)/2.) - (md-lo))
                r = math.ceil((len(wgts)/2.) + (hi-md))
                wgts = wgts[l:r]

                nk = np.sum(wgts*Cxk_slice,axis=1) + 0.001

                #SCORE                
                SRC_term = calc_full(nk, alpha_SRC, sum_SRC, lg_sum_SRC, sum_lg_SRC)
                BG_term = calc_full(nk, alpha_BG, sum_BG, lg_sum_BG, sum_lg_BG)
                score[row,col] = SRC_term - BG_term

                #score[row,col] where col = x and row = half_wd
                #so if score>0 (found source), source extends from [col-row:col+row+1]
                #compress to 1d binary array with 0s where score<=0; 1s where score>0
                #TODO adapt for more thresholds
                if score[row,col]>0:
                    score1d[col-row:col+row+1] =1

                #GRADIENT
                grad_m,grad_sigma = calc_gradients(np.arange(lo,hi+1),half_wd,md)
                w = calc_grad_weight(nk,alpha_SRC,alpha_BG,np.sum(nk),sum_BG,sum_SRC)
                gm = calc_fullgrad(w,Cxk_slice,grad_m)
                gs = calc_fullgrad(w,Cxk_slice,grad_sigma)

                gradients[row,col,1] = gm
                gradients[row,col,0] = gs

        print row

    #found_sources = mp.contiguous_regions(score1d==1)
    #TP=0;FP=0;FN=0

    scoresfile = outfile + '_scores.txt'
    np.savetxt(scoresfile,score)
    binsfile = outfile + '_mybins.txt'
    np.savetxt(binsfile,BINS)
    gradientfile = outfile + '_gradients.txt'
    with file(gradientfile,'w') as out:
       out.write('# Array shape: {0}\n'.format(gradients.shape))
       for dslice in gradients:
           np.savetxt(out,dslice)
           out.write('# New slice\n')


    Y,X = np.mgrid[0:max_wd,0:max_spread]
    U = gradients[:,:,0]
    V = gradients[:,:,1]
    plt.streamplot(X, Y, U, V,color='k')
    gradfig = gradientfile.split('.')[0]
    plt.savefig(gradfig)
    plt.clf()

    score1dfile = outfile + '_score1d'
    plt.plot(y,'k.')
    if opts.seed:
        plt.plot(shape1,'b-') 
        sigma1=shape1.copy()
        sigma1[0:left1]=0;sigma1[right1:sigma1.size]=0
        plt.plot(sigma1,'g.')
        plt.plot(shape2,'b-') 
        sigma2=shape2.copy()
        sigma2[0:left2]=0;sigma2[right2:sigma2.size]=0
        plt.plot(sigma2,'g.')
        plt.plot(shape3,'b-')
        sigma3=shape3.copy()
        sigma3[0:left3]=0;sigma3[right3:sigma3.size]=0
        plt.plot(sigma3,'g.')
    plt.plot(score1d,'r-')
    plt.savefig(score1dfile)
    plt.clf()

    #make average bins for the figure (TODO: plot all bins)
    if opts.dirichlet:
        BINS = np.mean(BINS,axis=0)
 
    #make the output figures
    if opts.nohisto:
        make_figs(x,y,BINS,outfile,score,gradients,histo=False)
    else:
        make_figs(x,y,BINS,outfile,score,gradients)



