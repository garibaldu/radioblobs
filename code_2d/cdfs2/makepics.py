import numpy as np
import pyfits
import os
import matplotlib.pyplot as plt

fitsfile = 'CDFS_map.fits'
hdulist = pyfits.open(fitsfile,memmap=True)
z = hdulist[0].data
hdulist.close()
while z.shape[0] == 1: 
    z = z[0]


r = z.shape[0] - 500
c = z.shape[1] - 500

for i in range(25):
    print i
    go = True
    while(go):
        row = np.random.randint(r)
        col = np.random.randint(c)
        zsmall=z[row:row+500,col:col+500]
        if (np.isnan(zsmall)).sum()==0:
            go=False
    out = '%s_%s' % (row,col)
    os.mkdir(out)
    out += '/data'
    with file(out, 'w') as outfile:
        outfile.write('# cdfs %s %s\n' % (row, col))
        np.savetxt(outfile, zsmall, fmt='%.20f')
    plt.clf()
    plt.imshow(zsmall,interpolation='nearest',cmap='gray',origin='lower')
    plt.clim(0,0.001)    
    plt.savefig(out)

