import numpy as np 
from scipy.stats import norm
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from numpy.linalg import pinv, inv
import json


class RBF():
    """
    Implements Radial Basis Function (RBF) network. 
    Uses KMeans clustering to compute centers if none are given. 
    """
    def __init__(self, n_centers = 8, activation = 'gaussian', sigma = 1, centers = None, lambdaReg = 0.):
        """
        n_centers = number of centers 
        activation = RBF activation function (can be gaussian, linear)  
        """
        self.set_params(n_centers, activation, sigma, centers, lambdaReg)
        
    # compatibility with sklearn 
    def set_params(self, n_centers = 8, activation = 'gaussian', sigma = 1, centers = None, lambdaReg = 0.):
        self.n_centers = n_centers
        self.activation = activation
        if activation == 'gaussian':
            self.sigma = sigma 
        self.centers = centers
        self.lambdaReg = lambdaReg
        return self

    def get_params(self, deep = False):
        result = {
            'n_centers'  : self.n_centers,
            'activation' : self.activation,
            'sigma'      : self.sigma,
            'centers'    : self.centers,
            'lambdaReg'  : self.lambdaReg
        }
        return result

    def fit(self, X, y):
        """
        Train RBF network on labelled data.
        Returns the weight vector for the trained network. 
        X = data 
        y = labels
        """
        # compute centers using kmeans
        if self.centers is None:
            kmeans = KMeans(n_clusters = self.n_centers)
            kmeans.fit(X)
            self.centers = kmeans.cluster_centers_
        # compute distances 
        D = cdist(X, self.centers)
        # activation 
        if self.activation == 'gaussian':
            Phi = norm.pdf(D, scale = self.sigma)
        if self.activation == 'linear':
            Phi = D
        # compute weight vector by invertion 
        lambdaEye = self.lambdaReg*np.eye(Phi.shape[1])
        self.w = inv(Phi.T@Phi + lambdaEye)@(Phi.T@y) # reduces to inv(Phi)@y when lambda = 0 
        return self.w
        
    def predict(self, X):
        """
        Applies the RBF function to data. 
        Returns vector of labels. 
        X = input 
        """
        # compute distances to centroids 
        D = cdist(X, self.centers)
        # apply activation function 
        if self.activation == 'gaussian':
            Phi = norm.pdf(D, scale = self.sigma)
        if self.activation == 'linear':
            Phi = D
        # answer is matrix multiplication by weight vector 
        result = np.dot(Phi, self.w)
        return result

    def score(self, X, y):
        yp = self.predict(X)
        return self._RMSE(y,yp)

    def _RMSE(self,y, yp):
        """
        Computes the root mean squared error (RMSE)
        """
        return np.sqrt(np.sum((yp-y)**2)/y.shape[0])  
    
    




