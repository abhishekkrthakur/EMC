import numpy as np

VERBOSE = True

class gaussian_params:

	def __init__ (self, mu, sigma):
		self.mu = np.array(mu).reshape((1,-1))
		self.sigma = np.array(sigma)
		if VERBOSE: 
			print "mu = ", self.mu
			print "sigma = ", self.sigma

	def proba(self, x):
		t1 = (1./np.sqrt(np.linalg.det(self.sigma)))
		t2 = np.exp(-(x - self.mu).dot(np.linalg.pinv(self.sigma)).dot((x-self.mu).T)/2.)
		prob = t1 * t2
		return prob.diagonal()

	def _var(self, sigma):
		delta = abs(sigma - self.sigma)
		self.sigma = sigma
		return np.sqrt(e.mean())/2

	def _mu(self, mu):
		delta = abs(self, mu)
		self.mu = mu 
		return delta.mean()/2
