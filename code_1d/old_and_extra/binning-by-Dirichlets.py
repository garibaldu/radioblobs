"""
Problem: For our approach we need the incoming real-valued intensity
values to be discretized or "binned", but there are 2 problems with
binning: (a) we don't have any informed basis for deciding on what bin
boundaries to use, and (b) discretization results in sudden changes at
the borderline values, which introduces variation into the system that
wasn't there in the original.

When a new value comes along, we might be better off including
"proportions of it" in several nearby bins, not just one. How to do
this in a sensible, robust way?

Idea: if we don't know which bins to use, maybe we should make up
heaps of them. Then for a given pixel intensity value x, we can have a
histogram of "possible" bin assignments instead of a single one.

The Dirichlet distribution gives a way to do this, since it produces a
set of numbers that sum to one, and parameters "alpha" enable one to
control both the typical widths of bins (our intuitions/whatever about
what their relative sizes should be) and their variation (ie. our
uncertainty about precisely where they should lie).

Is this useful? Hoping so.
This code shows the histograms as a thinking tool...........................
"""

import numpy as np
import numpy.random as rng
import pylab as pl
import sys
num_bins = 10
num_Dirichlets = 50


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print 'usage: python %s  STRETCH_FACTOR    TOTAL_ALPHA' %(sys.argv[0])
        print 'eg: python %s  1.5  5.0' %(sys.argv[0])
        sys.exit(0)

    STRETCH_FACTOR=float(sys.argv[1])     # Change this one and play........
    TOTAL_ALPHA = float(sys.argv[2])               # Change this one and play........

    alpha = np.ones((num_bins ),float) 
    # just trying exponential binning thing.................
    #Here each bin is getting bigger by factor.
    for i in range(1,len(alpha)): alpha[i] = STRETCH_FACTOR*alpha[i-1]   
    # And now set the overall scale of the alphas
    alpha = alpha * TOTAL_ALPHA / np.sum(alpha)
    print 'alphas are ',alpha, '\n-------------------------------------------------------------'


    internal_bin_borders  = np.cumsum(rng.dirichlet(alpha, num_Dirichlets),1)[:,:-1]
    # DONE! that's way too easy.. The rest of this script is just pretties.



    softbins = []
    test_step=0.008   # step size for the test points for the 2nd figure.
    test_vals = np.arange(0.0, 1.0, test_step)
    for x in test_vals:
        z = np.ravel(np.sum(x > internal_bin_borders, 1))
        counts, borders = np.histogram(z, bins=np.arange(num_bins+1))
        softbins.append(counts)
    softbins = np.array(softbins)

    # show stuff
    fig = pl.figure(figsize=(6,8.0))
    pl.subplot(211)

    vals = internal_bin_borders[:25,:]
    L,z = vals.shape
    print L,z
    for i in range(z):
        vv = vals[:,i]
        ii = range(len(vv))
        pl.plot(vv,ii,'-k',color='gray',linewidth=2)
    pl.axis([0.0,1.0, 0,24.5])
    pl.xlabel('pixel intensity')
    pl.ylabel('example bin borders')
    pl.yticks([])
    #pl.text(0,25,'(total alpha=%.1f, stretch=%.1f)' %(TOTAL_ALPHA, STRETCH_FACTOR))
    

    pl.subplot(212)
    pl.xlabel('pixel intensity')
    pl.ylabel('bin index')
    #pl.xticks([0,len(test_vals)-1],[test_vals[0],test_vals[len(test_vals)-1]]) # just start and end
    pl.yticks(range(num_bins))
    pl.imshow(np.transpose(softbins),interpolation='nearest',cmap='hot',origin='lower',aspect=6) # I picked that aspect completely by hand!!...... don't know what it means really
    pl.xticks([])
    pl.draw()
    outfile = 'test_DirBins_s%1.3f_a%3.1f.png' % (STRETCH_FACTOR,TOTAL_ALPHA)
    pl.savefig(outfile)
    print 'Wrote %s' % (outfile)





    # HERE I AM JUST TRYING OUT IDEAS FURTHER TO THE ABOVE, WITH A
    # FAKED EMPIRICAL DISTRIBUTION OF PIXEL VALUES. THROW THIS OUT
    # LATER....
    # these test_vals should really be a whole window's worth, from a real image!
    test_vals = np.power(rng.random((100,)), 5.0)

    """
    pl.clf()
    pl.subplot(211)
    pl.hist(test_vals,50)
    pl.xlabel('value')

    # NOW, I'm going to try to find an "optimal" stretch factor,
    # namely one that, given some particular set of example intensity
    # values, gives overall bin occupancies that are as close as
    # possible to uniform. I'll just do this by brute-force here...............
    possible_stretches = np.arange(1.0, 3.5, 0.1)
    entropy = np.zeros(possible_stretches.shape,float) # initialization
    for ind,stretch in enumerate(possible_stretches):
        # set the alphas
        alpha = np.ones((num_bins ),float) 
        for i in range(1,len(alpha)): alpha[i] = stretch*alpha[i-1]   
        alpha = alpha * TOTAL_ALPHA / np.sum(alpha)
        # invent the set of borders
        internal_bin_borders  = np.cumsum(rng.dirichlet(alpha, num_Dirichlets),1)[:,:-1]

        # find the overall occupancies, and calc the entropy of that.
        total_counts = np.zeros(num_bins, float)
        for x in test_vals:
            z = np.ravel(np.sum(x > internal_bin_borders, 1))
            counts, borders = np.histogram(z, bins=np.arange(num_bins+1))
            total_counts = total_counts + counts
        normed = total_counts/np.sum(total_counts)
        normed = normed + 0.0000001 # hack to avoid log(0). Has very little effect.
        entropy[ind] = -np.sum(np.dot(normed, np.log(normed)))
        print 'stretch: %.1f  gave entropy:  %.2f' %(stretch,entropy[ind])
    pl.subplot(212)
    pl.plot(possible_stretches,entropy,'ok')
    pl.xlabel('stretch factor')
    pl.ylabel('entropy')
    pl.draw()
    outfile = 'test_DirBins_s%1.3f_a%3.1f_entropies.png' % (STRETCH_FACTOR,TOTAL_ALPHA)
    pl.savefig(outfile)
    print 'Wrote %s' % (outfile)
    """
