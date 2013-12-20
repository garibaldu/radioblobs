import numpy as np
import numpy.random as rng
import pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mp 
from matplotlib.patches import Ellipse
import copy, sys
import math
import optparse
import scipy.signal
import scipy.special.basic as sp
import scipy.optimize as sop
import scipy
from numpy.core.umath_tests import inner1d


INCLUDE_BACKGROUND_COMPENSATION = False

source_to_background_ratio = np.log(0.1/0.9)

def sq(x):  # Am using this just to make the code more readable.
    return np.power(x,2.0)

def make_dirichlet_bins(data,num_bins,strategy,num_dirs=50,alpha=10.,stretch_factor=None,total_alpha=None,safety_gap=np.inf):

    z = copy.copy(data)
    z = np.ravel(z)
    z.sort()
    z = np.delete(z, np.where(np.isnan(z)))
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
    return mybins #return the bin borders


def make_bin_borders(data,num_bins,strategy='eqocc',safety_gap=np.inf,fname=None):
    z = copy.copy(data)
    z = np.ravel(z)
    z.sort()
    z = np.delete(z, np.where(np.isnan(z)))
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


def make_alphaBG(BINS,Nx,Ny,Z,dirichlet):

    if dirichlet:
        alpha_BGs = np.zeros((BINS.shape[0],BINS.shape[1]-1))
        K = BINS.shape[1]-1
        Cxk = np.zeros((Ny,Nx,K))
        for b in range(BINS.shape[0]):
            alpha_BGs[b] = np.histogram(np.ravel(Z),bins=BINS[b])[0]
            #for row in range(Ny):
            #    for col in range(Nx):
            #        Cxk[row,col] += np.histogram(Z[row,col],bins=BINS[b])[0]
            for i in range(K-1):
                Cxk[:,:,i]+=np.asarray((Z>=BINS[b,i])&(Z<BINS[b,i+1]),dtype=int)
            Cxk[:,:,K-1]+=np.asarray((Z>=BINS[b,K-1])&(Z<=BINS[b,K]),dtype=int)
        alpha_BG = np.mean(alpha_BGs,axis=0) + 1.0
        Cxk /= float(BINS.shape[0])
    else:
        alpha_BG = np.histogram(np.ravel(Z),bins=BINS)[0] + 1.0
        K = len(BINS)-1
        Cxk = np.zeros((Ny,Nx,K))
        #for row in range(Ny):
        #    for col in range(Nx):
        #        Cxk[row,col]=np.histogram(Z[row,col],bins=BINS)[0] 
        for i in range(K-1):
            Cxk[:,:,i]=np.asarray((Z>=BINS[i])&(Z<BINS[i+1]),dtype=int)
        Cxk[:,:,K-1]=np.asarray((Z>=BINS[K-1])&(Z<=BINS[K]),dtype=int)



    return Cxk, alpha_BG


def make_cxk(BINS,Nx,Ny,Z):

    K = len(BINS)-1
    Cxk = np.zeros((Ny,Nx,K))
    for i in range(K-1):
        Cxk[:,:,i]=np.asarray((Z>=BINS[i])&(Z<BINS[i+1]),dtype=int)
    Cxk[:,:,K-1]=np.asarray((Z>=BINS[K-1])&(Z<=BINS[K]),dtype=int)

    return Cxk


def get_BG(fname):
    """Get background alpha vector from LDA output"""
    CWT = np.delete(np.delete(np.genfromtxt(fname,comments='#'),0,1),0,0)
    #"biggest" topic is background (return this as background alpha vector)
    t0 = CWT[:,0]
    t1 = CWT[:,1]
    if np.sum(t0) > np.sum(t1):
        return t0, t1
    else:
        return t1, t0

# A useful method, used for making fake blobs, and for specifying a
# window profile for giving pixels different weights, when calculating
# the counts "in a window".
def make_gaussian_blob(X,Y,midx,midy,spreadx,spready,rotn=0.):
    xDist = (X-midx) # these are entire matrices
    yDist = (Y-midy)
    #if spreadx<spready:
    #    temp=spreadx; spreadx = spready; spready = temp;

    a = sq(math.cos(rotn))/(2.*sq(spreadx)) + sq(math.sin(rotn))/(2.*sq(spready))
    b = (-math.sin(2.*rotn)/(4*sq(spreadx))) + (math.sin(2*rotn)/(4*sq(spready)))
    c = sq(math.sin(rotn))/(2*sq(spreadx)) + sq(math.cos(rotn))/(2*sq(spready))

    W = np.exp(-1.0* (a*sq(xDist) + 2.*b*xDist*yDist + c*sq(yDist)))
    return W


############################ SCORE METHODS ############################

def calc_weighted_counts(theta, args):
    # WAS 'calc_params' but it didn't calc any params.
    # Passing it just theta and args now as simpler.

    mdx,mdy,sigmax,sigmay,phi = theta
    Cxk, X,Y,alpha_SRC,alpha_BG = args  
    wgts = make_gaussian_blob(X,Y,mdx,mdy,sigmax,sigmay,phi)

    N0, N1 = wgts.shape

    nkVec = (Cxk * wgts.reshape(N0,N1,1) ).sum(0).sum(0)
    return wgts, nkVec


def calc_logL(n, alphas):
    """ Calculate the log likelihood under DirMult distribution with alphas=avec, given data counts of nvec."""
    lg_sum_alphas = math.lgamma(alphas.sum())
    sum_lg_alphas = np.sum(scipy.special.gammaln(alphas))
    lg_sum_alphas_n = math.lgamma(alphas.sum() + n.sum())
    sum_lg_alphas_n = np.sum(scipy.special.gammaln(n+alphas))
    return lg_sum_alphas - sum_lg_alphas - lg_sum_alphas_n + sum_lg_alphas_n 


def calc_score(theta, args):
    """ Calculate and return the score  (WAS score_wrapper) """ 
    wgts, nkVec = calc_weighted_counts(theta, args)

    mdx,mdy,sigmax,sigmay,phi = theta
    Cxk, X,Y,alpha_SRC,alpha_BG = args

    logL_SRC = calc_logL(nkVec, alpha_SRC)
    logL_BG = calc_logL(nkVec, alpha_BG)
    score = logL_SRC - logL_BG #+ source_to_background_ratio

    if INCLUDE_BACKGROUND_COMPENSATION:
        nkVec_n = (alpha_BG/alpha_BG.sum()) * nkVec.sum()
        logL_compensation_SRC = calc_logL(nkVec_n, alpha_SRC)
        logL_compensation_BG = calc_logL(nkVec_n, alpha_BG)
        compensation_score = logL_compensation_SRC - logL_compensation_BG
        score = score - compensation_score

    return -score  # minus, since being fed to a minimizer not a maximizer


############################ GRADIENT METHODS ############################

def calc_df_dtheta(X,Y,midx,midy,spreadx,spready,rotation):
    """ Calculate gradients for given ellipse parameters
        Returns five window-sized arrays."""

    #W = make_gaussian_blob(X,Y,midx,midy,spreadx,spready,rotation)
    #if spreadx<spready:
    #    temp=spreadx; spreadx = spready; spready = temp;
    # These are just shorthand definitions.  I've renamed while checking. 
    # r is for rotation (as t/theta is taken).
    xDist, yDist = (X-midx), (Y-midy)
    xDist2, yDist2 = sq(xDist), sq(yDist)
    sigx2, sigy2 = sq(spreadx), sq(spready)
    cosr,sinr = math.cos(rotation), math.sin(rotation)
    cosr2,sinr2 = sq(cosr), sq(sinr)
    sin_2r,cos_2r = math.sin(2.*rotation), math.cos(2.*rotation)

    # These a,b,c seem okay...............
    a = cosr2/(2.*sigx2) + sinr2/(2.*sigy2)
    b = -(sin_2r/(4.*sigx2)) + (sin_2r/(4.*sigy2))   # nb. this could be simplified.
    c = sinr2/(2.*sigx2) + cosr2/(2.*sigy2)

    df_dmx = (2.*a*xDist + 2.*b*yDist)
    df_dmy = (2.*b*xDist + 2.*c*yDist)
    df_dsx = (xDist2*cosr2 - xDist*yDist*sin_2r + yDist2*sinr2)/(np.power(spreadx,3)) 
    df_dsy = (xDist2*sinr2 + xDist*yDist*sin_2r + yDist2*cosr2)/(np.power(spready,3))
    df_dr  = (((sinr*cosr)/(sigx2)-(sinr*cosr)/(sigy2))*xDist2 - 2.*((-cos_2r)/(2.*sigx2)+(cos_2r)/(2.*sigy2))*xDist*yDist - ((sinr*cosr)/(sigx2)-(sinr*cosr)/(sigy2))*yDist2)

    return np.asarray((df_dmx, df_dmy, df_dsx, df_dsy, df_dr))


def calc_Qdiff(nVec, alphaS, alphaB):
    """ Calculate Q_k - Q_base, for each bin k. Returns k-length vector."""
    N = nVec.sum()
    Qdiff = sp.psi(nVec + alphaS) - sp.psi(nVec+alphaB)  - sp.psi(N+alphaS.sum()) + sp.psi(N+alphaB.sum())
    return Qdiff 


def calc_logL_grad(Qdiff, Cxk_wgtd, dW_dtheta):
    """ Assembles the gradient of the score w.r.t. the parameters??
    Calculate full gradient: wgts * (data * grad) """  

    T0,T1,T2 = dW_dtheta.shape
    K0,K1,K2 = Cxk_wgtd.shape

    dg = (Cxk_wgtd.reshape(1,K0,K1,K2)) * (dW_dtheta.reshape(T0,T1,T2,1))
    dgs = dg.sum(1).sum(1)
    score_gradient = inner1d(Qdiff,dgs)

    return score_gradient


def calc_score_gradient(theta, args):
    """ Calculate and return the gradient """
    mdx,mdy,sigmax,sigmay,phi = theta
    Cxk,X,Y,alpha_SRC,alpha_BG = args

    wgts, nkVec = calc_weighted_counts(theta, args)
    N0,N1 = wgts.shape
    CtimesW = Cxk * wgts.reshape(N0,N1,1)

    df_dtheta = calc_df_dtheta(X,Y,mdx,mdy,sigmax,sigmay,phi)
    Qdiff = calc_Qdiff(nkVec,alpha_SRC,alpha_BG)
    score_grad = calc_logL_grad(Qdiff,CtimesW,df_dtheta)

    if INCLUDE_BACKGROUND_COMPENSATION:
        nkVec_compensation = (alpha_BG/alpha_BG.sum()) * np.sum(nkVec)
        Qdiffn = calc_Qdiff(nkVec_compensation,alpha_SRC,alpha_BG)
        score_grad -= calc_logL_grad(Qdiffn,CtimesW,df_dtheta)

    return -score_grad  # minus, because it's used by a minimizer not a maximizer


if __name__ == "__main__":

    parser = optparse.OptionParser(usage="usage %prog [options]")

    parser.add_option("-n","--numbins",type = "int",dest = "K",default=0,
                      help="number of bins (ignored if strategy is dexpocc or fromfile)")
    parser.add_option("-b","--bins_fname",dest = "bfname",
                      help="bin borders filename")   
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
        outfile += '_%s' % seed
        # make an "image"
        rng.seed(seed)  # seed the random number generator here
        noise_size=1.0

        Nx, Ny = 50, 70 #number of pixels in a fake test image
        X,Y = np.meshgrid(np.arange(Nx), np.arange(Ny))  # these are both matrices of indices, each the same size as image

        # make up the 'shapes' of the sources
        mid1x, mid2x, mid3x = rng.random()*Nx, rng.random()*Nx, rng.random()*Nx
        mid1y, mid2y, mid3y = rng.random()*Ny, rng.random()*Ny, rng.random()*Ny
        print 'Random sources placed at %.0d,%.0d; %.0d,%.0d; %.0d,%.0d ' %(mid1x,mid1y,mid2x,mid2y,mid3x,mid3y)
        # length scale
        spread1x = 1+2*rng.random()
        spread1y = 1+2*rng.random()  
        spread2x = 1+2*rng.random()
        spread2y = 1+2*rng.random() 
        spread3x = 1+2*rng.random()
        spread3y = 1+2*rng.random()   
        print 'with (x,y) sigmas: %.1f,%.1f; %.1f,%.1f; %.1f,%.1f ' %(spread1x,spread1y,spread2x,spread2y,spread3x,spread3y)

        rot1,rot2,rot3 = rng.random()*(2.*math.pi),rng.random()*(2.*math.pi),rng.random()*(2.*math.pi)
        print 'and rotation variables: %s; %s; %s ' % (rot1,rot2,rot3)

        shape1 = make_gaussian_blob(X,Y,mid1x,mid1y,spread1x,spread1y,rot1) *3.0
        shape2 = make_gaussian_blob(X,Y,mid2x,mid2y,spread2x,spread2y,rot2) *3.0
        shape3 = make_gaussian_blob(X,Y,mid3x,mid3y,spread3x,spread3y,rot3) *3.0

        # noise character of sources
        variance = np.abs(noise_size*(1.0 - shape1 + shape2)) # source 3 has no variance effect
        noise = rng.normal(0.,variance)

        # mean_intensity character of sources
        mean = shape1 + shape2 + shape3
        Z = mean + noise

        #outobjs = 'r%s_objects' % seed
        #plt.imshow(mean,interpolation='nearest',cmap='gray',origin='lower')
        #plt.savefig(outobjs)

        outobjs = 'r%s_objects-noise' % seed
        plt.imshow(Z,interpolation='nearest',cmap='gray',origin='lower')
        plt.savefig(outobjs)


    else:    # it's not a digit, so it's a filename. File should be just list of numbers.
        infile = opts.infile
        Z = np.genfromtxt(infile)
        Nx,Ny = Z.shape
        X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))  # these are both matrices of indices, each the same size as image
        mean = Z.copy()
        outfile += '_%s' % infile


    #make bins (here, from the naked image)
    if opts.dirichlet:
        outfile += '_dirichletborders'
        BINS = make_dirichlet_bins(Z,K,strategy)
        if K == 0:
            K = BINS.shape[1] - 1        # mean_intensity character of sources
        print 'Note: an example overall histogram: (using the first of the dirichlet histograms)'
        print np.histogram(np.ravel(Z),bins=BINS[0])[0]
    else:
        BINS = make_bin_borders(Z,K,strategy,safety_gap=np.inf,fname=opts.bfname)
        if K == 0:
            K =  len(BINS) - 1
        print 'Note: this makes the overall histogram this: (reality-check the final one especially)'
        print np.histogram(np.ravel(Z),bins=BINS)[0]

    outfile += '_K%d' % K

    #get background alphas from LDA output, if specified
    if opts.CWT:
        alpha_BG, alpha_SRC = get_BG(opts.CWT)
        outfile += '_LDA'
        Cxk = make_cxk(BINS,Nx,Ny,Z)
    else: 
        print 'making bins and binning data ...'
        # bogus, but we're setting the background alphas as if there were
        # no sources in the image at the moment....
        Cxk, alpha_BG = make_alphaBG(BINS,Nx,Ny,Z,opts.dirichlet)
        # 1.0 to be agnostic: all multinomial distributions are equally likely to be drawn
        alpha_SRC = 1.0 * np.ones(alpha_BG.shape)  

    #k=np.arange(float(K))
    #CMAP = pl.cm.RdBu
    #plt.clf()
    #plt.imshow(np.sum((Cxk*k),axis=2).reshape(Ny,Nx),interpolation='nearest',cmap=CMAP,origin='lower')
    #out = outfile + '_inbins'
    #plt.savefig(out)





    print 'proceeding to gradient descent\n'

    #do gradient ascent
    num_top = 3
    num_iters = 50
    top_scores=np.zeros((num_top,11))
    messages = np.zeros(9)

    Bounds = [(0,Nx),(0,Ny),(1,Nx/2.),(1,Ny/2.),(0,2.*math.pi)]

    print "gradient descent ... "

    for i in range(num_top):

        optima = np.zeros((num_iters,6))
        for j in range(num_iters):
            print '------------------------------------------\niter %s.%s\n' % (i,j)
            brightr, brightc = np.where(Z==np.nanmax(Z))
            idx = rng.randint(brightr.size)
            mdx = brightc[idx]
            mdy = brightr[idx]
            sigmax = (Nx/15.)+rng.rand()*(Nx/10.)
            sigmay = (Ny/15.)+rng.rand()*(Ny/10.)
            phi = rng.rand()*2.*math.pi

            theta = [mdx,mdy,sigmax,sigmay,phi]

            args = [Cxk, X,Y,alpha_SRC, alpha_BG]

            sltn, its, rc = sop.fmin_tnc(calc_score, theta, calc_score_gradient, [args], bounds=Bounds, fmin=-1e10, maxfun=1000, accuracy=1e-16)
            sc = calc_score(sltn, args)
            optima[j,:5] = sltn
            optima[j,5] = -sc
            messages[rc] += 1

        top_opt = scipy.delete(optima, np.where(np.isnan(optima)), 0)
        top_opt = top_opt[np.argsort(top_opt[:,5])][-1]
        top_scores[i,:6] = top_opt
        #remove best source
        top_mdx =top_opt[0]
        top_mdy =top_opt[1]
        top_sigx=top_opt[2]
        top_sigy=top_opt[3]
        top_phi =top_opt[4]


        theta = [top_mdx,top_mdy,top_sigx,top_sigy,top_phi]
        grad = -calc_score_gradient(theta, args)
        top_scores[i,6:] = grad
        print grad

        x = (X-top_mdx)*math.cos(top_phi)+(Y-top_mdy)*math.sin(top_phi)
        y = -(X-top_mdx)*math.sin(top_phi)+(Y-top_mdy)*math.cos(top_phi)
        a = sq(2*top_sigx) 
        b = sq(2*top_sigy) 
        Z[np.where((sq(x)/a+sq(y)/b) <= 1)] = np.nan 
        if np.isnan(Z).sum() == Nx*Ny: break
        if opts.dirichlet:
            BINS = make_dirichlet_bins(Z,K,strategy)
        else:
            BINS = make_bin_borders(Z,K,strategy,safety_gap=np.inf,fname=opts.bfname)
        Cxk, alpha_BG = make_alphaBG(BINS,Nx,Ny,Z,opts.dirichlet)
        #out = outfile + '_%dth_src' % i
        #masked_array = np.ma.array(Z, mask=np.isnan(Z))
        #cmap = pl.cm.gray
        #cmap.set_bad('r',1.)
        #plt.clf()
        #plt.imshow(Z,interpolation='nearest',cmap=cmap,origin='lower')
        #plt.savefig(out)
        #plt.clf()
        #plt.imshow(np.sum((Cxk*k),axis=2).reshape(Ny,Nx),interpolation='nearest',cmap=CMAP,origin='lower')
        #out = outfile + '_inbins-%dth' % i
        #plt.savefig(out)
        if top_opt[5] <= 0: break #quit early if only background left


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

    plt.imshow(mean,interpolation='nearest',cmap='gray',origin='lower')
    plt.plot(top_scores[:,0],top_scores[:,1],'rx')
    for i in range(top_scores.shape[0]):
        top_mdx =top_scores[i,0]
        top_mdy =top_scores[i,1]
        top_sigx=top_scores[i,2]
        top_sigy=top_scores[i,3]
        top_phi =top_scores[i,4] 
        rect = Ellipse((top_mdx,top_mdy),top_sigx*2,top_sigy*2,top_phi*(180./math.pi),edgecolor='red',facecolor='green',alpha = 0.5)
        pl.gca().add_patch(rect)
    plt.ylim(0,Ny-1)
    plt.xlim(0,Nx-1)


    plt.savefig(outfile)

    np.savetxt(outfile,top_scores[:,:5])

    out_gt = outfile + '_GT'
    gt = np.zeros((3,5))
    gt[0]=mid1x,mid1y,spread1x,spread1y,rot1
    gt[1]=mid2x,mid2y,spread2x,spread2y,rot2
    gt[2]=mid3x,mid3y,spread3x,spread3y,rot3
    np.savetxt(out_gt,gt)

    #print gt
    #print top_scores[:,:2]


