import numpy as np
import numpy.random as rng
import pylab as pl
import copy

N = 1000 #number of pixels
width = 160.0  # length scale
noise_size=1.0
K=10
lk = np.log(K)




def eqocc(data,num_bins,safety_gap=np.inf):
    z = copy.copy(data)
    z.sort()
    top, bottom = z[-1], z[0]
    mybins = []
    step = len(z)/num_bins
    for i in range(0,len(z)-step+1,step):
        mybins.append(z[i])
    mybins.append(z[-1])
    mybins[-1] += safety_gap
    mybins[0]  -= safety_gap

    return mybins



# make an "image"
start = rng.random() * (N-width)
x = np.arange(N)
truth = np.zeros(x.shape)
truth[start:start+width] = 0.5
noise = rng.normal(0,noise_size,x.shape)
#make bins on noise (uniform bins)
BINS = eqocc(noise,K)
y = truth + noise
#y = (y-np.min(y))/(np.max(y)-np.min(y)) + np.abs(rng.normal(0,0.01,x.shape))


fig = pl.figure(figsize=(10,8))
pl.subplot(511)
pl.plot(x,y,'.k')
pl.subplot(512)
pl.plot(x,truth,'k',linestyle='dashed',linewidth=5,alpha=0.3)


width_h = int(width/2)
width_d = int(width*2)




score = np.zeros(N-width)

mids = range(int(width/2),int(N-width/2),1)
for i,m in enumerate(mids):

    nk = np.histogram(y[m-width/2:m+width/2],bins=BINS)[0]
    nk = nk + 0.01
    Ns = np.sum(nk)
    pk = nk/Ns
    ent = -np.sum(pk * np.log(pk))

    score[i] = Ns * (lk - ent)

pl.subplot(513)
pl.plot(mids,score,'-r',linewidth=2,alpha=0.5)


score = np.zeros(N-width_h)

mids = range(int(width_h/2),int(N-width_h/2),1)
for i,m in enumerate(mids):

    nk = np.histogram(y[m-width_h/2:m+width_h/2],bins=BINS)[0]
    nk = nk + 0.01
    Ns = np.sum(nk)
    pk = nk/Ns
    ent = -np.sum(pk * np.log(pk))

    score[i] = Ns * (lk - ent)

pl.subplot(514)
pl.plot(mids,score,'-g',linewidth=2,alpha=0.5)


score = np.zeros(N-width_d)

mids = range(int(width_d/2),int(N-width_d/2),1)
for i,m in enumerate(mids):

    nk = np.histogram(y[m-width_d/2:m+width_d/2],bins=BINS)[0]
    nk = nk + 0.01
    Ns = np.sum(nk)
    pk = nk/Ns
    ent = -np.sum(pk * np.log(pk))

    score[i] = Ns * (lk - ent)
    #print score[n]

pl.subplot(515)
pl.plot(mids,score,'-y',linewidth=2,alpha=0.5)


pl.savefig('DirModelComparison.png')
