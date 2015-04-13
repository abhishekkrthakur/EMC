import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd

class KMeans(object):
	def __init__(self, X, num_clusters, niter, delta):
		self.X = X
		self.num_clusters = num_clusters
		self.niter = niter
		self.delta = delta

	def random_sampling(self):
		"""
		Helper function for the k-Means clustering.
		Returns given number of random samples from the data.
		"""
		randomIndex = np.random.randint(0, self.X.shape[0], self.num_clusters)
		print randomIndex
		return self.X[randomIndex, :]

	def distance_function(self, X,Y):
		return cdist(X, Y, 'euclidean')

	def fit(self):
		initial = self.random_sampling()
		
		N, dim = self.X.shape
		k, cdim = initial.shape
		if dim != cdim:  
			raise ValueError("Error! Centers must have same number of columns as the data!")

		print initial

		all_X = np.arange(N)
		old_dist = 0
		for jiter in xrange(1, self.niter + 1):
			dist = self.distance_function(self.X, initial)
			xtoc = dist.argmin(axis = 1)
			distances = dist[all_X, xtoc]
			avgdist = distances.mean()

			if (1-self.delta) * old_dist <= avgdist <= old_dist or jiter==self.niter:
				break
			old_dist = avgdist

			for jc in range(k):
				c = np.where(xtoc == jc)[0]
				if len(c) > 0:
					initial[jc] = np.mean(self.X[c])

		return initial, xtoc, distances 

if __name__ == '__main__':
	from sklearn import datasets
	iris = datasets.load_iris()
	X = iris.data[:, :2]  # we only take the first two features.
	Y = iris.target
	#print data
	#data = np.array(data)[:,1:]
	#from sklearn.decomposition import PCA
	#pca = PCA(n_components=2)
	#data = pca.fit_transform(data)
	km = KMeans(X, num_clusters=2, niter=100, delta=0.0001) 
	initial, xtoc, distances = km.fit()
	print initial
	print xtoc
	import matplotlib.pyplot as plt
	plt.scatter(X[:,0], X[:,1])
	plt.scatter(initial[:,0], initial[:,1], color = 'r')
	plt.show()
