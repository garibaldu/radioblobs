import numpy as np
import matplotlib.pyplot as plt

dirs_a = np.zeros((10,3))
dirs_b = np.zeros((10,3))
dirs_c = np.zeros((10,3))

for i in range(10):

  a = np.random.dirichlet([1,1,1])
  b = np.random.dirichlet([10,10,10])
  c = np.random.dirichlet([2,5,20])
  dirs_a[i] = a
  dirs_b[i] = b
  dirs_c[i] = c

  lefts = [1,2,3]
  f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
  f.set_figwidth(13)
  f.set_figheight(3)
  plt.ylim(0,1)
  ax1.set_xlim(0.8,4)
  ax1.bar(lefts, a)
  ax2.bar(lefts, b)
  ax3.bar(lefts, c)
  for a in f.axes: a.set_xlim(0.8,4)
  for a in f.axes: a.axis("off")
  plt.subplots_adjust(wspace=.8,left=0.05,right=0.95)

  plt.savefig('dirichlet_draws_%s' % i)


plt.figure()
ind = np.arange(10)    # the y locations for the groups
height = 0.5       # the width of the bars: can also be len(x) sequence
plt.xlim(0,1)
plt.barh(ind, dirs_a[:,0], height, color='r')
plt.barh(ind, dirs_a[:,1], height, color='g', left=dirs_a[:,0])
plt.barh(ind, dirs_a[:,2], height, color='b', left=np.sum([dirs_a[:,0],dirs_a[:,1]],axis=0))
plt.axis("off")
plt.savefig("stacked_a")

plt.figure()
plt.xlim(0,1)
plt.barh(ind, dirs_b[:,0], height, color='r')
plt.barh(ind, dirs_b[:,1], height, color='g', left=dirs_b[:,0])
plt.barh(ind, dirs_b[:,2], height, color='b', left=np.sum([dirs_b[:,0],dirs_b[:,1]],axis=0))
plt.axis("off")
plt.savefig("stacked_b")

plt.figure()
plt.xlim(0,1)
plt.barh(ind, dirs_c[:,0], height, color='r')
plt.barh(ind, dirs_c[:,1], height, color='g', left=dirs_c[:,0])
plt.barh(ind, dirs_c[:,2], height, color='b', left=np.sum([dirs_c[:,0],dirs_c[:,1]],axis=0))
plt.axis("off")
plt.savefig("stacked_c")
