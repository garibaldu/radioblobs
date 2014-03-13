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

def make_figs(x,y,BINS,outfile,score,gradients,histo=True):
    """
    making the figures
    """

    # tedious but necessary definitions for the axes.
    fig1 = pl.figure(1,figsize=(7,4),dpi=300)
    if (histo):
        left, width, space = 0.1, 0.6, 0.03
        bottom, height, rawheight, cmapwidth = 0.05, 0.6, 0.2, 0.03
    else:
        left, width, space = 0.1, 0.8, 0.03
        bottom, height, rawheight, cmapwidth = 0.05, 0.6, 0.2, 0.0
    cmapheight = height
    fiddle = 0.052
    rect_raw                = [left+fiddle, bottom+height+rawheight/2+2*space, width-fiddle*2, rawheight/2]
    rect_raw_in_bins = [left+fiddle, bottom+height+space, width-fiddle*2, rawheight/2]
    rect_scores = [left, bottom, width, height]
    rect_cmap   = [left+width+space, bottom, cmapwidth, height/2]

    if (histo): 
        rect_histo = [left+width+space, bottom+height+space, 0.15, rawheight/2]
        rect_histo_upper = [left+width+space, bottom+height+rawheight/2+2*space, 0.15, rawheight/2]
    axScores = pl.axes(rect_scores)
    axRaw = pl.axes(rect_raw)
    axRawInBins = pl.axes(rect_raw_in_bins)
    if (histo): 
        axCmap = pl.axes(rect_cmap)
        axHisto = pl.axes(rect_histo)
        axHistoUpper = pl.axes(rect_histo_upper)
    cmap = pl.cm.RdBu   # if you want a different map (jet, hot, cool, etc) PuOr and RdGy have the appeal of being white in the middle, which can be made score=zero.  Looks like RdGy would be awesome if it went backwards.............

    axRawInBins.plot(x, np.digitize(y,BINS),'s',markersize=2,markeredgecolor='None',alpha=0.3)
    axRawInBins.axis('off')
    axRawInBins.set_ylabel('bins')
    axRawInBins.axis([0,N,0,len(BINS)])

    axRaw.plot(x,y,'.k',markersize=4)
    axRaw.set_ylabel('values')
    axRaw.axis('off')
    for b in BINS:
        axRaw.plot([0,N],[b,b],'-b',alpha=0.35)
    miny,maxy = np.min(y),np.max(y)
    gap = maxy-miny
    axRaw.axis([0,N,np.min(y)-gap/20,np.max(y)+gap/20])

    im2 = axScores.imshow(score,interpolation='nearest',origin='lower',cmap=cmap)
    axScores.axis('off')
    #axScores.set_xlabel('position')
    #axScores.set_ylabel('half width')

    #STREAMPLOT ...
    Y,X = np.mgrid[0:N/2.,0:N]
    U = gradients[:,:,1]
    V = gradients[:,:,0]
    lw = np.power(U,2.0) + np.power(V,2.0)
    lw = 5*lw/lw.max()
    axScores.streamplot(X, Y, U, V,color='k',density=4.,linewidth = 0.5)#linewidth=lw)
    """
    """


    #min_colorscore, max_colorscore = np.min(np.ravel(score)),np.max(np.ravel(score)) # just calc these for the colormap limits
    min_colorscore, max_colorscore = -10.0, 10.0
    norm = mpl.colors.Normalize(vmin=min_colorscore, vmax=max_colorscore) # the extent of the colormap
    im2.set_norm(norm) 
    axScores.axis([0,N,0,N/2])

 
    #COLOURBAR
    if (histo): 
        mpl.colorbar.ColorbarBase(axCmap, cmap=cmap, norm=norm, orientation='vertical')
        pl.gcf().text(left+width+cmapwidth+3*space, bottom, 'background',color=cmap(0))
        pl.gcf().text(left+width+cmapwidth+3*space, bottom+ cmapheight/2, 'source', color=cmap(255))
    

    # reset the extreme bin limits just to get the histogram pic right.
    if histo:
        histobins = BINS
        histobins[0], histobins[-1] = np.min(y), np.max(y)
        n, histobins, ppatches = axHisto.hist(y, histobins, normed=0, orientation='horizontal',histtype='bar',color='gray')
        axHisto.axis([0,np.max(n),np.min(y),np.max(y)])
        axHisto.axis('off')

        n, histobins, ppatches = axHistoUpper.hist(np.digitize(y,BINS), len(BINS), normed=0, orientation='horizontal',histtype='bar',color='k',alpha=0.3)
        axHistoUpper.axis([0,np.max(n),np.min(y)+1,np.max(y)])
        axHistoUpper.axis('off')


    pl.savefig(outfile,dpi=300)
    print 'Wrote %s.png' % (outfile)


def make_dirichlet_bins(data,num_bins,strategy,num_dirs=50,alpha=10.,stretch_factor=None,total_alpha=None,safety_gap=np.inf):
#TODO: random seed?
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
                mybins[d,p+1] = bottom + props[p] * datarange

    elif strategy == 'expocc':
        print "strategy expocc not implemented for dirichlet bins yet"
        num_datapts = z.size
        for d in range(dirs.shape[0]/2):
            props = (np.cumsum(dirs[d])*num_datapts)[:-1]
            for p in range(len(props)):
                mybins[d,p+1] = (z[props[p]] + z[props[p]+1])/2
        datarange = top - bottom
        for d in range(dirs.shape[0]/2,dirs.shape[0]):
            props = np.cumsum(dirs[d])[:-1]
            for p in range(len(props)):
                mybins[d,p+1] = props[p] * datarange
        #sys.exit(-1)

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

def calc_lgamma_vect(vect):
    """Calculate the log gamma of each number in a vector """
    v = np.array(vect) #+ 1e-10
    if np.any(v<0.5): print 'FOUND ELEMENT <0.5 \n %s \n' % v
    for i in range(v.size):       
        v[i] = math.lgamma(v[i])
    return v

def calc_full(n, alphas, sum_alphas, lg_sum_alphas, sum_lg_alphas):
    """ Calculate the log likelihood under DirMult distribution with alphas=avec, given data counts of nvec."""
    lg_sum_alphas_n = math.lgamma(sum_alphas + np.sum(n))
    sum_lg_alphas_n = np.sum(calc_lgamma_vect(n+alphas))
    s = lg_sum_alphas - sum_lg_alphas - lg_sum_alphas_n + sum_lg_alphas_n
    #s = sum_lg_alphas_n - lg_sum_alphas_n
    return s

def calc_gradients(x,sigma,m):
    """ Calculate gradients for m and sigma for a given m, sigma, and window[xposl:xposr]
        Returns two x-length vectors."""
    wx = np.exp(-((np.power((m-x),2.))/(2.*np.power(sigma,2.))))

    grad_m = (wx*(x-m))/(np.power(sigma,2.))
    grad_sigma = (wx*(np.power((m-x),2.)))/(np.power(sigma,3.))

    return grad_m, grad_sigma

def calc_grad_weight(nks, alphaS, alphaB, N, AB, AS):
    """ Calculate the weights for each bin k. Returns k-length vector."""
    K = nk.size
    w = sp.psi(nks + alphaS) - sp.psi(nks+alphaB) + sp.psi(N+AB) - sp.psi(N+AS)
    return w

def calc_fullgrad(wgt,data,gradient):
    
    full = np.dot(wgt, (np.sum(data*gradient,axis=1) ))
    return full


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
    parser.add_option("-c",action="store_true",dest="doCorrection",default=False,
                      help="make our bogus small-sample correction to score")
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
    K = opts.K

    if opts.seed:
        seed = opts.seed
        # make an "image"
        rng.seed(seed)  # seed the random number generator here

        N = 600 #number of pixels in a fake test image
        noise_size=1.0
        x = np.arange(N)

        # make up the 'shapes' of the sources
        mid1, mid2, mid3, mid4 = rng.random() * N,rng.random() * N,rng.random() * N,rng.random() * N
        print 'Random sources placed at ',mid1, mid2, mid3, mid4
        spread1 = int(2 + rng.random()*N/100)  # length scale skinny
        spread2 = int(N/20+2*rng.random())  # length scale
        spread3 = int(N/50+2*rng.random())  # length scale
        spread4 = 5
        print 'Random sources have spreads ',spread1,spread2,spread3,spread4
        shape1 = np.exp(-0.5*np.power((x-mid1)*1.0/spread1,2.0))
        shape2 = np.exp(-0.5*np.power((x-mid2)*1.0/spread2,2.0))
        shape3 = np.exp(-0.5*np.power((x-mid3)*1.0/spread3,2.0))
        shape4 = np.exp(-0.5*np.power((x-mid3)*1.0/spread3,2.0))
        # noise character of sources
        variance = 1.0 + 3*shape1 -0.75*shape2 # source 3 has no variance effect
        #variance = variance + x/float(len(x)) # to mimic steady change over large scales
        noise = noise_size * rng.normal(0,variance,x.shape)
        # mean_intensity character of sources
        mean = 3* shape1 + 1*shape2 + 0.5*shape3 + 100*shape4
        y = mean + noise


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

    outfile = 'DMR'
    outfile += '_%d' % seed
    outfile += '_%s' % strategy
    outfile += '_K%d' % K
    if (opts.doCorrection): outfile += '_corrected'
    if opts.dirichlet:   outfile += '_dirichletborders'

    #get background alphas from LDA output, if specified
    if opts.CWT:
        alpha_BG = get_BG(opts.CWT)
        outfile += '_LDA'
    else:
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
    alpha_SRC[0] = 0.5
    for i in range(1,len(alpha_SRC)):
        alpha_SRC[i] = 1.1*alpha_SRC[i-1]
    print 'alpha_SRC: ',alpha_SRC

    max_spread = N
    max_wd = max_spread/2
    score = np.zeros((max_wd,max_spread))
    gradients = np.zeros((max_wd,max_spread,2))
    grad_ests = np.zeros((max_wd,max_spread,2))
    grad_diff = np.zeros((max_wd,max_spread,2))


    #1st two terms for full calculation
    sum_BG = np.sum(alpha_BG)
    lg_sum_BG = math.lgamma(sum_BG)
    sum_lg_BG = np.sum(calc_lgamma_vect(alpha_BG))

    sum_SRC = np.sum(alpha_SRC)
    lg_sum_SRC = math.lgamma(sum_SRC)
    sum_lg_SRC = np.sum(calc_lgamma_vect(alpha_SRC))
      

    if opts.hard:
        outfile += '_hardborders'

    sig_wd = 3.
    for half_wd in range(1,max_wd,1): # wd is width of the window
        row = half_wd
        for col in range(max_spread):
            # evaluate the score of a model that has middle=row, spread=col.
            md = float(col)

            lo = max(0, md - sig_wd*half_wd)
            hi = min(N-1, md + sig_wd*half_wd)
            
            if ((hi - lo)>1) and (md >= lo) and (md <= hi):
                # otherwise it's a fairly silly model!
                bound = y[lo:hi+1]
                Cxk_slice = Cxk[:,lo:hi+1]

                if opts.hard:
                    win_size = half_wd*(2.*sig_wd) + 1.
                    wgts = np.zeros((win_size))
                    wgts[win_size/2.-half_wd:win_size/2.+half_wd+1] += 1.
                    l = math.floor((len(wgts)/2.) - (md-lo))
                    r = math.ceil((len(wgts)/2.) + (hi-md))
                    wgts = wgts[l:r]
                else:
                    #wgts = scipy.signal.gaussian(win_size,half_wd)
                    xb = np.arange(lo,hi+1,dtype='float')
                    wgts = np.exp(-(np.power((xb-md),2.)/(2.*np.power(half_wd,2.))))

                nk = np.sum(wgts*Cxk_slice,axis=1) 

                #SCORE                
                SRC_term = calc_full(nk, alpha_SRC, sum_SRC, lg_sum_SRC, sum_lg_SRC)
                BG_term = calc_full(nk, alpha_BG, sum_BG, lg_sum_BG, sum_lg_BG)
                sc = SRC_term - BG_term 
                sc -= np.log(10.0) # this is the effect of our prior on P(source)
                

                #GRADIENT
                grad_m,grad_sigma = calc_gradients(xb,half_wd,md)
                w = calc_grad_weight(nk,alpha_SRC,alpha_BG,np.sum(nk),sum_BG,sum_SRC)
                gm = calc_fullgrad(w,Cxk_slice,grad_m)
                gs = calc_fullgrad(w,Cxk_slice,grad_sigma)

                # Marcus attempt to do the "correction"
                if (opts.doCorrection):
                    nkcorr = alpha_BG/np.sum(alpha_BG) * np.sum(nk) # IS RIGHT? SEEMS OK.
                    # nk_n = (alpha_BG/sum_BG) * np.sum(nk) was earlier one that worked...

                    # correct the score...
                    Scorrector = calc_full(nkcorr, alpha_SRC, sum_SRC, lg_sum_SRC, sum_lg_SRC)
                    Bcorrector = calc_full(nkcorr, alpha_BG, sum_BG, lg_sum_BG, sum_lg_BG)
                    sc = sc - (Scorrector-Bcorrector)

                    # correct the gradient too...
                    grad_m,grad_sigma = calc_gradients(xb,half_wd,md)
                    w = calc_grad_weight(nkcorr,alpha_SRC,alpha_BG,np.sum(nkcorr),sum_BG,sum_SRC)
                    gmcorr = calc_fullgrad(w,Cxk_slice,grad_m)
                    gscorr = calc_fullgrad(w,Cxk_slice,grad_sigma)
                    gm = gm - gmcorr
                    gs = gs - gscorr

                # store the result in the matrix.
                score[row,col] = sc
                gradients[row,col,1] = gm
                gradients[row,col,0] = gs


        if (row % 10 == 0): print row

    #TP=0;FP=0;FN=0
    writetxtfiles = False


    #make average bins for the figure (TODO: plot all bins)
    if opts.dirichlet:
        BINS = np.mean(BINS,axis=0)

    #make the output figures
    if opts.CWT:
        make_figs(x,y,BINS,outfile,score,gradients,histo=False)
    elif opts.nohisto:
        make_figs(x,y,BINS,outfile,score,gradients,histo=False)
    else:
        make_figs(x,y,BINS,outfile,score,gradients)



