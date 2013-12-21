"""
This makes a series of fake images, by adding Gaussian noise to a base
level that has local areas that are squared exponential shaped
"hills".  The "ground truth" (hills) are stored in one figure, the
images in another, and a third shows the inferred density, done by
computing the value of the KL(Q||P) score.

Marcus Frean. 2012.
"""
import sys
import numpy as np, numpy.random as rng
import pylab as pl, matplotlib.cm as cm


def do_exit():
    print ('usage: python %s  [image_size  num_sources  noise_size  output_image_name]' % (sys.argv[0]))
    sys.exit('eg: python %s 60 3 2.0 mytestimage' % (sys.argv[0]))


def doOne(subplot_num, n_sources):
    truth = np.zeros(X1.shape)
    for i in range(n_sources):
        midx1,midx2 = rng.random() * N,  rng.random() * N
        intensity = rng.random()
        expsum =np.power((X1-midx1)*1.0/sigma,2.0) + np.power((X2-midx2)*1.0/sigma,2.0)
        truth = truth + intensity * np.exp(-0.5*expsum)
    noise = rng.normal(0,noise_size,X1.shape)
    y = truth + noise
    y = (y-np.min(y))/(np.max(y)-np.min(y)) + np.abs(rng.normal(0,0.01,X1.shape))


    # Evaluate the reversed KL score, throughout the image
    KLRevscore = -100000 * np.ones(X1.shape)
    KLscore = -100000 * np.ones(X1.shape)
    for midx1 in range(N):
        for midx2 in range(N):
            # make up a "model"
            expsum =np.power((X1-midx1)*1.0/model_sigma,2.0) + np.power((X2-midx2)*1.0/model_sigma,2.0)
            model = np.exp(-0.5*expsum)

            # score the model against the "image"
            KLRevscore[midx1][midx2] = np.sum(np.ravel(model*np.log(y)))/np.sum(np.ravel(model))

    # rescale to range in [0,1]
    KLRevscore = (KLRevscore-np.min(KLRevscore))/(np.max(KLRevscore)-np.min(KLRevscore))

    
    # draw the ground truth (as an image), 
    pl.figure('ground')
    pl.subplot(3,1,subplot_num)
    pl.imshow(truth,interpolation='nearest',cmap=cm.gray)
    pl.title('ground truth %s' % (subplot_num))

    # draw the noisy image.
    pl.figure('image')
    pl.subplot(3,1,subplot_num)
    pl.imshow(y,interpolation='nearest',cmap=cm.gray)
    pl.title('image %s' % (subplot_num))

    # draw the inferred source density (as contours).
    pl.figure('contours')
    pl.subplot(3,1,subplot_num)
    pl.imshow(truth*0.0,interpolation='nearest',cmap=cm.gray)
    CS = pl.contour(X2,X1,KLRevscore,5,linewidths=np.arange(5),
                 colors=((.2,.2,0),(.4,.4,0),(.6,.6,0),(.8,.8,0),(1,1,0)))
    pl.clabel(CS, inline=1, fontsize=10)
    pl.title('inferred density %s' % (subplot_num))



sigma = 10            # length scale
model_sigma = sigma  # Beware the inverse criminal!!!!

if __name__ == "__main__":

    if len(sys.argv) < 1:
        do_exit()
    if len(sys.argv) == 1 :
        N, noise_size, num_sources, image_name = 60, 1.0, 2, 'test'
    else: # extra arguments are being supplied
        if len(sys.argv) < 5: do_exit()
        N = int(sys.argv[1])  # image size
        num_sources = int(sys.argv[2])
        noise_size =  float(sys.argv[3])
        image_name = sys.argv[4]

    # make an "image"
    x1 = np.arange(N)
    x2 = np.arange(N)
    X1, X2 = np.meshgrid(x1, x2)

    pl.figure('ground',figsize=(5,15))
    pl.figure('image',figsize=(5,15))
    pl.figure('contours',figsize=(5,15))

    for i in range(1,4):
        n = rng.randint(1,num_sources+1)
        doOne(i,n)

    for figname in ('ground','image','contours'):
        pl.figure(figname)
        pl.draw()
        pl.savefig(image_name+'_'+figname)
        print 'Wrote %s.png' % (image_name+'_'+figname)
#        pl.show()
