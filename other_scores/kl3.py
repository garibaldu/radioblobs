import sys
import numpy as np, numpy.random as rng
import pylab as pl, matplotlib.cm as cm


def do_exit():
    print ('usage: python %s  [image_size  num_sources  noise_size  output_image_name]' % (sys.argv[0]))
    sys.exit('eg: python %s 100 3 2.0 mytestimage' % (sys.argv[0]))

def calc_score_everywhere(model_sigma):
    KLRevscore = -100000 * np.ones(X1.shape)
    for midx1 in range(N):
        for midx2 in range(N):
            # make up a "model" centered at this site
            expsum =np.power((X1-midx1)*1.0/model_sigma,2.0) + np.power((X2-midx2)*1.0/model_sigma,2.0)
            model = np.exp(-0.5*expsum)
            # score the model against the "image"
            KLRevscore[midx1][midx2] = np.sum(np.ravel(model*np.log(y)))/np.sum(np.ravel(model))
    # rescale to range in [0,1]
    KLRevscore = (KLRevscore-np.min(KLRevscore))/(np.max(KLRevscore)-np.min(KLRevscore))
    return KLRevscore



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
    sigma = 10.0    # length scale of blobs in 'truth'

    fig = pl.figure(figsize=(15,10))

    # make an "image"
    x1 = np.arange(N)
    x2 = np.arange(N)
    X1, X2 = np.meshgrid(x1, x2)

    truth = np.zeros(X1.shape)
    for i in range(num_sources):
        midx1,midx2 = rng.random() * N,  rng.random() * N
        expsum =np.power((X1-midx1)*1.0/sigma,2.0) + np.power((X2-midx2)*1.0/sigma,2.0)
        truth = truth + np.exp(-0.5*expsum)
    noise = rng.normal(0,noise_size,X1.shape)
    y = truth + noise
    y = (y-np.min(y))/(np.max(y)-np.min(y)) + np.abs(rng.normal(0,0.01,X1.shape))


    # draw the ground truth (as an image), 
    pl.subplot(241)
    pl.imshow(truth,interpolation='nearest',cmap=cm.gray)
    pl.title('ground truth')

    # draw the noisy image.
    pl.subplot(242)
    pl.imshow(y,interpolation='nearest',cmap=cm.gray)
    pl.title('image')

    s=3
    for sig in (sigma*0.2, sigma*0.5, sigma, sigma*2.0, sigma*4.0):
        # Evaluate the reversed KL score, throughout the image
        KLRevscore = calc_score_everywhere(sig)
        # draw the inferred source density (as contours).
        pl.subplot(2,4,s+1)
        s += 1
        pl.imshow(truth*0.0,interpolation='nearest',cmap=cm.gray)
        CS = pl.contour(X2,X1,KLRevscore,5,linewidths=np.arange(5),
                        colors=((.2,.2,0),(.4,.4,0),(.6,.6,0),(.8,.8,0),(1,1,0)))
        pl.clabel(CS, inline=1, fontsize=10)
        pl.title('score with sigma=%.1f' % (sig))

    pl.draw()
    fig.savefig(image_name)
    print 'Wrote %s.png' % (image_name)
    pl.show()
