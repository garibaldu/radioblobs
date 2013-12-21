import numpy as np
import numpy.random as rng
import pylab as pl

N = 1000 #number of pixels
sigma = 30.0  # length scale
model_sigma = np.arange(sigma/3, 3*sigma, sigma/3) # length scales
noise_size=1.0
alphalevel = 0.5

# make an "image"
mid = rng.random() * N
x = np.arange(N)
truth = 0.5*np.exp(-0.5*np.power((x-mid)*1.0/sigma,2.0))
noise = rng.normal(0,noise_size,x.shape)
y = truth + noise
y = (y-np.min(y))/(np.max(y)-np.min(y)) + np.abs(rng.normal(0,0.01,x.shape))


fig = pl.figure(figsize=(10,4))
pl.subplot(211)
pl.plot(x,y,'.k')
pl.subplot(212)
pl.plot(x,truth,'k',linestyle='dashed',linewidth=5,alpha=0.3)
KLscore = np.ones(N)
KLRevscoreA = np.ones(N)
KLRevscoreB = np.ones(N)
KLRevscoreC = np.ones(N)

for scale in model_sigma:
    for mid in range(N):
        # make up a "model"
        model = np.exp(-0.5*np.power((x-mid)*1.0/scale,2.0)) + 0.00000001
        
        # score the model against the "image"
        KLscore[mid] = np.sum(y*np.log(model))/np.sum(model)
        KLRevscoreA[mid] = np.sum(model*np.log(y))/np.sum(model)
        KLRevscoreB[mid] = np.sum(model*np.power(y,2.0))/np.sum(model)
        KLRevscoreC[mid] = np.sum(model*y)/np.sum(model)

    KLscore = (KLscore-np.min(KLscore))/(np.max(KLscore)-np.min(KLscore))
    KLRevscoreA = (KLRevscoreA-np.min(KLRevscoreA))/(np.max(KLRevscoreA)-np.min(KLRevscoreA))
    KLRevscoreB = (KLRevscoreB-np.min(KLRevscoreB))/(np.max(KLRevscoreB)-np.min(KLRevscoreB))
    KLRevscoreC= (KLRevscoreC-np.min(KLRevscoreC))/(np.max(KLRevscoreC)-np.min(KLRevscoreC))
#pl.plot(x,KLscore,'-y',linewidth=3)
    pl.plot(x,KLRevscoreA,'-r',linewidth=1,alpha=alphalevel)
    pl.plot(x,KLRevscoreB,'-b',linewidth=1,alpha=alphalevel)
    pl.plot(x,KLRevscoreC,'-y',linewidth=2,alpha=alphalevel)

pl.draw()
fig.savefig('testKL')
print 'wrote testKL'
