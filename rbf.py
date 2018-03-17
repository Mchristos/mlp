import numpy as np 
from scipy.stats import norm
from scipy.spatial.distance import cdist
from numpy.linalg import pinv
import json


class RBF():
    def __init__(self, centers, activation = 'gaussian', sigma = 1, w = None):
        """
        centers = RBF centers
        activation = RBF activation function (e.g. gaussian, linear)  
        w = trained weights 
        """
        self.w = None # learned weights 
        self.c = centers
        self.activation = activation
        if activation == 'gaussian':
            self.sigma = sigma 
        


    def train(self, X, y):
        """
        Train RBF network on labelled data using given centers and activation function. 
        Returns the weight vector for trained network. 
        X = data 
        y = labels
        c = centres 
        f_a = activation function
        """
        D = cdist(X, self.c)
        if self.activation == 'gaussian':
            Phi = norm.pdf(D, scale = self.sigma)
        if self.activation == 'linear':
            Phi = D
        self.w = np.dot(pinv(Phi), y)
        return self.w
        
    def predict(self, X):
        """
        Applies the RBF function to data. 
        Returns vector of outputs. 
        X = input 
        """
        # compute distances to centroids 
        D = cdist(X, self.c)
        # apply activation function 
        if self.activation == 'gaussian':
            Phi = norm.pdf(D, scale = self.sigma)
        if self.activation == 'linear':
            Phi = D
        # answer is matrix multiplication by weight vector 
        result = np.dot(Phi, self.w)
        return result

    def serialize(self, path):
        selfobj = {}
        selfobj['weights'] = self.w.tolist()
        selfobj['centers'] = self.c.tolist()
        with open(path, 'w') as f:
            f.write(json.dumps(selfobj))
