import numpy as np

class EMClustering(object):

	def __init__(self, X, num_clusters = 2):
		self.X = X
		num_samples = X.shape[0]
		num_features = X.shape[1]
		self.num_clusters = num_clusters

	def gaussian_clusters(self):
		idx = np.random.choice(xrange(num_samples), self.num_clusters, replace = False)