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
This code shows the histograms as a thinking tool....................................
"""

import numpy as np
import numpy.random as rng
import pylab as pl
import sys
np.set_printoptions(precision=3,suppress=True) # print floats not too long.
num_bins = 10
num_Dirichlets = 500
np.set_printoptions(precision=4,suppress=True) # print floats to certain precision.


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print 'usage: python %s  STRETCH_RATIO    TOTAL_ALPHA' %(sys.argv[0])
        print 'STRETCH_RATIO is the ratio of occupancy in the LAST bin to that in the FIRST bin'
        print 'eg: python %s  0.01 5.0' %(sys.argv[0])
        sys.exit(0)
    else: print ''

    STRETCH_RATIO=float(sys.argv[1])     # Change this one and play........
    TOTAL_ALPHA = float(sys.argv[2])               # Change this one and play........

    alpha = np.ones((num_bins ),float) 
    #Here each bin is getting bigger by a factor that we figure out
    #from the desired RATIO of the first bin's occupancy to the last
    #one.
    STRETCH_FACTOR = np.power(STRETCH_RATIO, 1.0/(num_bins-1))
    for i in range(1,len(alpha)): alpha[i] = STRETCH_FACTOR*alpha[i-1]   
    # And now set the overall scale of the alphas
    alpha = alpha * TOTAL_ALPHA / np.sum(alpha)
    print 'STRETCH_RATIO = %f for %d bins --> STRETCH_FACTOR = %f\n' %(STRETCH_RATIO, num_bins,STRETCH_FACTOR)
    print 'alphas are ',alpha, '\n-------------------------------------------------------------'


    # these test_vals should really be a whole window's worth, from a real image!
    test_vals = np.power(rng.random((1000,)), 5.0)
    test_vals = test_vals*22 - 1  # just push them around a bit: shouldn't matter at all.
    num_test_vals = len(test_vals)

    # And how about Anna's idea: use Dir with a stretch factor of 1
    # (say...), but instead of interpreting the values as bin borders
    # themselves, let's take them to be break points in an overall ordering
    # of the example data we have. We should perhaps place the actual
    # bin border (for each Dirichlet's proposal) mid-way between the
    # corresponding points in the ranking.

    # So let's try this on the test_vals. First we sort them.
    sorted_vals = np.sort(test_vals)

    # invent the set of borders - a lovely numpy one-liner!
    internal_borders_in_rank  = (num_test_vals-1) * np.cumsum(rng.dirichlet(alpha, num_Dirichlets),1)[:,:-1] 
    # nb. we are ignoring the last col, as it's always 1 anyway by cumsum defn.



    # ---------- the conversion to actual bin borders -----------------------------------------
    # Now convert them.... I will be dumb and do via a big loop, but
    # what the heck! It's only done once anyway.
    internal_bin_borders = np.zeros(internal_borders_in_rank.shape, float) # space...
    for row,vec in enumerate(internal_borders_in_rank):
        inds_above = np.minimum(np.ceil(vec).astype(np.integer), num_test_vals-1)
        inds_below = np.maximum(np.floor(vec).astype(np.integer), 0)
        vals_above = sorted_vals[inds_above]
        vals_below = sorted_vals[inds_below]
        # I will interpolate linearly between the two "nearest neighbours"
        vals = vals_below + (vals_above - vals_below)*(vec - np.floor(vec)) 
        internal_bin_borders[row] = vals

    # ---------- rest is just pretties --------------------------------------------------------------------
    print 'a few internal_borders_in_rank, straight out of the Dirichlet:'
    print internal_borders_in_rank[:4]
    print ''
    print 'a few internal_bin_borders derived from applying those to test_vals:'
    print internal_bin_borders[:4]
    print ''

    # Finally, let's check the overall bin occupancies under this scheme.
    total_counts = np.zeros(num_bins, int)
    for x in test_vals:
        z = np.ravel(np.sum(x > internal_bin_borders, 1))
        counts, borders = np.histogram(z, bins=np.arange(num_bins+1))
        total_counts = total_counts + counts
    print 'an example of bin attributions, for an input of %.3f: ' %(x)
    print counts # just the last one in the random order of test_vals...
    print '\n total bins counts over all the test_vals: \n', total_counts
