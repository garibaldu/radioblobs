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


def sigm(t):
    return 1.0/(1.0+ np.exp(-t))

def skew_gen_gauss(x,mid):

    dev = x - mid
    beta, alpha = (5.-0.5)*rng.random()+0.5, 8.*rng.random()+6.
    ggd = beta/(2*alpha*math.gamma(1.0/beta)) * np.exp(-np.power(np.abs(dev)/alpha, beta))
    shape = ggd * sigm(rng.normal()*dev)
    height = (5-0.5)*rng.random()+0.5
    shape = height * shape/shape.max()

    return shape


f, ((ax1, ax2),
    (ax3, ax4),
    (ax5, ax6),
    (ax7, ax8),
    (ax9, ax10)) = plt.subplots(5,2, sharex='col',sharey='row')

axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10]

for seed in range(10):

    ax = axes[seed]
    ax.set_ylim(-6,6)
    ax.set_yticks([-6,0,6])
    #ax.set_xticks([0,250,500])

    # make an "image"
    rng.seed(seed+1)  # seed the random number generator here

    N = 500 #number of pixels in a fake test image
    noise_size=1.0
    x = np.arange(N)
    # make up the 'shapes' of the sources
    mid1, mid2, mid3 = rng.random() * N,rng.random() * N,rng.random() * N
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

    ax.plot(y,'k,')
    ax.plot(shape1,'b-')
    ax.plot(shape2,'r-')
    ax.plot(shape3,'g-')
    ax.plot([0]*N,'k-')

plt.savefig('1d-data-egs')
