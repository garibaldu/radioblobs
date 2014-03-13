import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import pylab as pl
from matplotlib.patches import Ellipse
import pyfits


if len(sys.argv) == 4:
    row = int(sys.argv[1])
    col = int(sys.argv[2])
    stub = sys.argv[3]
else:
    sys.exit('usage: python %s row col stub' % (sys.argv[0]))


fitsfile = '%s_map.fits' % stub

hdulist = pyfits.open(fitsfile,memmap=True)
z = hdulist[0].data
hdulist.close()
while z.shape[0] == 1: z = z[0]
zsmall=z[row:row+500,col:col+500]

try:
    gt = np.genfromtxt('%s_%s/%s_%s_%s_gt_FULL' % (row,col,stub,row,col))
except:
    sys.exit()#'no sources here: %s %s' %(row,col) )
gt = gt.clip(min=0)
if len(gt.shape)==1: gt = gt.reshape(1,gt.shape[0])
if int(row) in [2298,2743,2773,3051,6252,6412,6418,6430]:
    fnd = np.genfromtxt('%s_%s/DirModel_width_data_K10_cut' % (row,col))
else: 
    fnd = np.genfromtxt('%s_%s/DirModel_width_data_K10' % (row,col))
if len(fnd.shape)==1: fnd = fnd.reshape(1,fnd.shape[0])
fnd = np.delete(fnd,np.where(fnd[:,5]<=-np.log(0.01/0.99)),0)

TP = 0.
FP = 0.
FN = float(gt.shape[0])
prop = 0.1

foundouts = []

Nx,Ny = 500,500
X,Y = np.meshgrid(np.arange(Nx),np.arange(Ny))

for i in range(fnd.shape[0]):
    #print i

    if (fnd[i] == [0]*fnd.shape[1]).sum() == fnd.shape[1]: continue

    
    #Ze = np.zeros((Ny,Nx))
    #x =  (X-fnd[i,1])*math.cos(fnd[i,4])+(Y-fnd[i,0])*math.sin(fnd[i,4])
    #y = -(X-fnd[i,1])*math.sin(fnd[i,4])+(Y-fnd[i,0])*math.cos(fnd[i,4])
    #a = np.power((1.5*fnd[i,3]),2)
    #b = np.power((1.5*fnd[i,2]),2)
    #Ze[np.where((np.power(x,2)/a+np.power(y,2)/b) <=1)] = 1
    #ly = max(fnd[i,1]-1.5*fnd[i,3],0)
    #ry = min(fnd[i,1]+1.5*fnd[i,3],Ny)
    #lx = max(fnd[i,0]-1.5*fnd[i,2],0)
    #rx = min(fnd[i,0]+1.5*fnd[i,2],Nx)
    #Ze[lx:rx,ly:ry] =1

    x,y = fnd[i,0],fnd[i,1]

    found = False
    for j in range(gt.shape[0]):
        if (gt[j] == [0]*gt.shape[1]).sum() == gt.shape[1]: continue

        Zt = np.zeros((Ny,Nx))
        Zt[gt[j,4]:gt[j,5]+1,gt[j,2]:gt[j,3]+1] = 1

        xl,xr = gt[j,4],gt[j,5]+1
        yl,yr = gt[j,2],gt[j,3]+1

        #print np.logical_and(Ze==1, Zt==1).sum()

        #if np.logical_and(Ze==1, Zt==1).sum() >= prop*np.sum(Ze==1) and np.logical_and(Ze==1, Zt==1).sum() >= prop*np.sum(Zt==1):
        if x>=xl and x<=xr and y>=yl and y<=yr:
            TP += 1; FN -= 1; found = True
            #print fnd[i]
            #print gt[j]
            gt[j] = [0]*gt.shape[1]
            foundouts.append(i)
            break

    if not found: FP += 1


print '%s %s %s %s %s' % (TP, FP, FN,TP/(TP+FP),TP/(TP+FN))
#print 'ests: %s; trues: %s; precision: %s; recall: %s' % (fnd.shape[0],gt.shape[0],TP/(TP+FP),TP/(TP+FN))

#outf = '%s_%s/%s_%s_%s_cut' % (row,col,stub,row,col)
#out=np.asarray([TP,FP,FN,fnd.shape[0],gt.shape[0],TP/(TP+FP),TP/(TP+FN)])
#np.savetxt(outf, out)


#plt.clf()
#plt.imshow(zsmall,interpolation='nearest',cmap='gray',origin='lower')

#for i in range(len(foundouts)):
#    idx = foundouts[i]
#    top_mdx =fnd[idx,0]
#    top_mdy =fnd[idx,1]
#    top_sigx=fnd[idx,2]
#    top_sigy=fnd[idx,3]
#    top_phi =fnd[idx,4] 
#    a = top_sigx; b = top_sigy;
#    rect = Ellipse((top_mdx,top_mdy),a*2,b*2,top_phi*(180./math.pi),edgecolor='red',facecolor='green',alpha = 0.5)
#    pl.gca().add_patch(rect)


    #rect = Rectangle((gt[i,4],sublist[i,2]),sublist[i,5]-sublist[i,4],sublist[i,3]-sublist[i,2], edgecolor='red', facecolor='green', alpha=0.5)
    #pl.gca().add_patch(rect)
#plt.xlim(0,500)
#plt.ylim(0,500)
#plt.clim(0,0.0001)
#plt.savefig('%s_%s/%s_%s_%s_ew_TP' % (row,col,stub,row,col))


