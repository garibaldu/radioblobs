import numpy as np
import math


Nx,Ny = 50,70
X,Y = np.meshgrid(np.arange(Nx),np.arange(Ny))


for prop in range(5,100,10):

    prop /= 100.
    TP = 0
    FP = 0
    FN = 0

    for seed in range(50):

        #estimated sources
        estfname = 'DirModel_eqocc_%s_K10' % (seed+1)
        est = np.genfromtxt(estfname)

        Ze0 = np.zeros((Ny,Nx))
        x =  (X-est[0,0])*math.cos(est[0,4])+(Y-est[0,1])*math.sin(est[0,4])
        y = -(X-est[0,0])*math.sin(est[0,4])+(Y-est[0,1])*math.cos(est[0,4])
        a = np.power((2*est[0,2]),2)
        b = np.power((2*est[0,3]),2)
        Ze0[np.where((np.power(x,2)/a+np.power(y,2)/b) <=1)] = 1

        Ze1 = np.zeros((Ny,Nx))
        x =  (X-est[1,0])*math.cos(est[1,4])+(Y-est[1,1])*math.sin(est[1,4])
        y = -(X-est[1,0])*math.sin(est[1,4])+(Y-est[1,1])*math.cos(est[1,4])
        a = np.power((2*est[1,2]),2)
        b = np.power((2*est[1,3]),2)
        Ze1[np.where((np.power(x,2)/a+np.power(y,2)/b) <=1)] = 1

        Ze2 = np.zeros((Ny,Nx))
        x =  (X-est[2,0])*math.cos(est[2,4])+(Y-est[2,1])*math.sin(est[2,4])
        y = -(X-est[2,0])*math.sin(est[2,4])+(Y-est[2,1])*math.cos(est[2,4])
        a = np.power((2*est[2,2]),2)
        b = np.power((2*est[2,3]),2)
        Ze2[np.where((np.power(x,2)/a+np.power(y,2)/b) <=1)] = 1

    
        #ground truth
        gtfname = 'DirModel_eqocc_%s_K10_GT' % (seed+1)
        gt = np.genfromtxt(gtfname)
    
        Zg0 = np.zeros((Ny,Nx))
        x =  (X-gt[0,0])*math.cos(gt[0,4])+(Y-gt[0,1])*math.sin(gt[0,4])
        y = -(X-gt[0,0])*math.sin(gt[0,4])+(Y-gt[0,1])*math.cos(gt[0,4])
        a = np.power((2*gt[0,2]),2)
        b = np.power((2*gt[0,3]),2)
        Zg0[np.where((np.power(x,2)/a+np.power(y,2)/b) <=1)] = 1
    
        Zg1 = np.zeros((Ny,Nx))
        x =  (X-gt[1,0])*math.cos(gt[1,4])+(Y-gt[1,1])*math.sin(gt[1,4])
        y = -(X-gt[1,0])*math.sin(gt[1,4])+(Y-gt[1,1])*math.cos(gt[1,4])
        a = np.power((2*gt[1,2]),2)
        b = np.power((2*gt[1,3]),2)
        Zg1[np.where((np.power(x,2)/a+np.power(y,2)/b) <=1)] = 1
    
        Zg2 = np.zeros((Ny,Nx))
        x =  (X-gt[2,0])*math.cos(gt[2,4])+(Y-gt[2,1])*math.sin(gt[2,4])
        y = -(X-gt[2,0])*math.sin(gt[2,4])+(Y-gt[2,1])*math.cos(gt[2,4])
        a = np.power((2*gt[2,2]),2)
        b = np.power((2*gt[2,3]),2)
        Zg2[np.where((np.power(x,2)/a+np.power(y,2)/b) <=1)] = 1
    

        #count up overlaps
        count = 0
        if np.logical_and(Ze0==1, Zg0==1).sum() >= prop*np.sum(Ze0==1) and np.logical_and(Ze0==1, Zg0==1).sum() >= prop*np.sum(Zg0==1): count +=1
        if np.logical_and(Ze0==1, Zg1==1).sum() >= prop*np.sum(Ze0==1) and np.logical_and(Ze0==1, Zg1==1).sum() >= prop*np.sum(Zg1==1): count +=1
        if np.logical_and(Ze0==1, Zg2==1).sum() >= prop*np.sum(Ze0==1) and np.logical_and(Ze0==1, Zg2==1).sum() >= prop*np.sum(Zg2==1): count +=1
        if np.logical_and(Ze1==1, Zg0==1).sum() >= prop*np.sum(Ze1==1) and np.logical_and(Ze1==1, Zg0==1).sum() >= prop*np.sum(Zg0==1): count +=1
        if np.logical_and(Ze1==1, Zg1==1).sum() >= prop*np.sum(Ze1==1) and np.logical_and(Ze1==1, Zg1==1).sum() >= prop*np.sum(Zg1==1): count +=1
        if np.logical_and(Ze1==1, Zg2==1).sum() >= prop*np.sum(Ze1==1) and np.logical_and(Ze1==1, Zg2==1).sum() >= prop*np.sum(Zg2==1): count +=1
        if np.logical_and(Ze2==1, Zg0==1).sum() >= prop*np.sum(Ze2==1) and np.logical_and(Ze2==1, Zg0==1).sum() >= prop*np.sum(Zg0==1): count +=1    
        if np.logical_and(Ze2==1, Zg1==1).sum() >= prop*np.sum(Ze2==1) and np.logical_and(Ze2==1, Zg1==1).sum() >= prop*np.sum(Zg1==1): count +=1
        if np.logical_and(Ze2==1, Zg2==1).sum() >= prop*np.sum(Ze2==1) and np.logical_and(Ze2==1, Zg2==1).sum() >= prop*np.sum(Zg2==1): count +=1

        TP += count
        FP += 3 - count
        FN += 3 - count

    print '--------PROP: %s---------' % prop
    print TP 
    print FP
    print FN
    print float(TP)/150.


