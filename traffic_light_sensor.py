from matplotlib import pyplot as plt
import numpy as np 
from scipy.spatial.distance import cityblock


def knn(x_train,y_train,xt,k=17):
	labels = []
	for ix in range(x_train.shape[0]):
		d = cityblock(x_train[ix],xt)
		labels.append([d,y_train[ix]])
	sorted_labels = sorted(labels,key=lambda z:z[0])
	neighbours = np.asarray(sorted_labels)[:k,-1]
	# filtered_labels = filter(lambda z: z[0] <= 6.5,labels)
	# filtered_labels = np.asarray(filtered_labels)[:,-1]
	# print filtered_labels
	freq = np.unique(neighbours,return_counts=True)
	return freq[0][freq[1].argmax()]

#random distribution of traffic over two roads
dist_01 =  7*np.random.random_sample((50,)) - 4,4*np.random.random_sample((50,)) + 2 
dist_01 = np.asarray(dist_01)
dist_01=dist_01.T
dist_02 =  4*np.random.random_sample((50,)) + 3.5,7*np.random.random_sample((50,)) - 4.5

dist_02 = np.asarray(dist_02)
dist_02 = dist_02.T

r = dist_01.shape[0] + dist_02.shape[0]
c = dist_01.shape[1] + 1
data = np.zeros((r,c))
data[:dist_01.shape[0],:2] = dist_01
data[dist_02.shape[0]:,:2] = dist_02
data[:dist_01.shape[0],-1] = 1.0
data[dist_01.shape[0]:,-1] = 2.0
np.random.shuffle(data)
traffic_light = [6.5,6] # let us suppose that the traffic light is at point (6.5,6)

l=knn(data[:,:2],data[:,-1],traffic_light) # find the nearest neighbour cars from the trrafic light  
if l == 1.0:
	print "Traffic light for road 1 is switched on"
else:
	print 'Traffic Light for road 2 is switched'


#plot of traffic in the two roads
plt.xlim(-5,10)
plt.ylim(-5,10)
plt.scatter(dist_01[:,:1],dist_01[:,1:],color='c',marker='s',alpha=0.8)
plt.scatter(dist_02[:,:1],dist_02[:,1:],color='m',marker='^',alpha=0.8)
plt.scatter(6.5,6,color='g',marker='X')
plt.show()


