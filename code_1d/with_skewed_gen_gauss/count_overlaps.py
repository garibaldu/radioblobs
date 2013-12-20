import numpy as np
import matplotlib.pyplot as plt

TP = 0
FP = 0
FN = 0


xs = np.arange(0.05,1.0,0.05)
binstrat = ['width','width','eqocc','eqocc']
dirs = ['','dirichletborders_','','dirichletborders_']

w = np.zeros((19))
wd = np.zeros((19))
e = np.zeros((19))
ed = np.zeros((19))

arrays = [w,wd,e,ed]

for i in range(4):

    for prop in range(5,100,5):
        prop /= 100.
        TP = 0

        for seed in range(50):
            #if seed+1 in [4,5,8,9,13,16,20,22,24,30,34,39,43,49]: continue

            #estimated sources
            estfname = 'DirModel_%s_r%s_%sK10_fullscore_optima' % (binstrat[i],seed+1,dirs[i])
            est = np.genfromtxt(estfname)
            est=est[:,:2]

            est0_left  = est[0,0] - 2*est[0,1]
            est0_right = est[0,0] + 2*est[0,1]
            est0=np.arange(np.round(est0_left),np.round(est0_right)+1)    

            est1_left  = est[1,0] - 2*est[1,1]
            est1_right = est[1,0] + 2*est[1,1]
            est1=np.arange(np.round(est1_left),np.round(est1_right)+1)

            est2_left  = est[2,0] - 2*est[2,1]
            est2_right = est[2,0] + 2*est[2,1]
            est2=np.arange(np.round(est2_left),np.round(est2_right)+1)

            #ground truth
            gtfname = 'DirModel_%s_r%s_%sK10_fullscore_optima_GT' % (binstrat[i],seed+1,dirs[i])
            f = file(gtfname,'r')
            r = f.readlines()
            f.close()
    
            gt0m=float(r[0].strip().strip('#').split()[0])
            gt0s=float(r[0].strip().strip('#').split()[1])
            gt0_left  = gt0m - 2*gt0s
            gt0_right = gt0m + 2*gt0s
            gt0=np.arange(np.round(gt0_left),np.round(gt0_right)+1)
    
            gt1=np.asarray(r[502:1002],dtype='float')
            gt1=np.where(gt1>=0.05*gt1.max())[0]
        
            gt2=np.asarray(r[1003:1502],dtype='float')
            gt2=np.where(gt2>=0.05*gt2.max())[0]

            #count up overlaps
            count = 0
            if len(set(gt0).intersection(set(est0)))>=prop*len(gt0) and len(set(gt0).intersection(set(est0)))>=prop*len(est0): count +=1
            if len(set(gt0).intersection(set(est1)))>=prop*len(gt0) and len(set(gt0).intersection(set(est1)))>=prop*len(est1): count +=1
            if len(set(gt0).intersection(set(est2)))>=prop*len(gt0) and len(set(gt0).intersection(set(est2)))>=prop*len(est2): count +=1
            if len(set(gt1).intersection(set(est0)))>=prop*len(gt1) and len(set(gt1).intersection(set(est0)))>=prop*len(est0): count +=1
            if len(set(gt1).intersection(set(est1)))>=prop*len(gt1) and len(set(gt1).intersection(set(est1)))>=prop*len(est1): count +=1
            if len(set(gt1).intersection(set(est2)))>=prop*len(gt1) and len(set(gt1).intersection(set(est2)))>=prop*len(est2): count +=1
            if len(set(gt2).intersection(set(est0)))>=prop*len(gt2) and len(set(gt2).intersection(set(est0)))>=prop*len(est0): count +=1
            if len(set(gt2).intersection(set(est1)))>=prop*len(gt2) and len(set(gt2).intersection(set(est1)))>=prop*len(est1): count +=1
            if len(set(gt2).intersection(set(est2)))>=prop*len(gt2) and len(set(gt2).intersection(set(est2)))>=prop*len(est2): count +=1
    
            TP += count
    
        arrays[i][np.round((prop-0.05)/0.05)] = TP/150.


plt.plot(xs,w,'-',linewidth=2)
plt.plot(xs,wd,':',linewidth=2)
plt.plot(xs,e,'-.',linewidth=2)
plt.plot(xs,ed,'--',linewidth=2)
plt.legend(('Equal width','Dirichlet equal width','Equal occupancy','Dirichlet equal occupancy'),loc=3)
plt.xlim(0,1)
plt.ylim(0,1.2)
plt.xlabel('Threshold')
plt.ylabel('Precision / Recall')
plt.savefig('test')
print w
print wd
print e
print ed
