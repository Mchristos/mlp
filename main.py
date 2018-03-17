import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rbf import RBF
import json
from sklearn.externals import joblib
import time

def error(yp, y):
    """
    Compute error between predicted yp and actual y values. 
    """
    return np.sqrt(sum((yp-y)**2)/len(y))


def train(X, y):
    """ train rbf model, save model"""
    startTime = time.time()
    print("training...")
    # hyperparameters    
    n_pca_components = 10
    n_centers = 500
    activation_sigma = 10

    # pre-process (PCA)
    X = (X - np.mean(X))/np.std(X, axis=0)
    pca = PCA(n_components = n_pca_components)
    pca.fit(X)
    Xt = pca.transform(X)

    # cluster data to get centres 
    kmeans = KMeans(n_clusters = n_centers)
    kmeans.fit(Xt)
    centers = kmeans.cluster_centers_

    # train RBF 
    rbf = RBF(centers, sigma=activation_sigma)
    rbf.train(Xt, y) # dim = #centers 

    # save model 
    joblib.dump(rbf, 'rbf.pkl')
    joblib.dump(pca, 'pca.pkl')

    print("took %0.3f seconds" % (time.time() - startTime))

def predict(X):
    # print("predicting...")

    rbf = joblib.load('rbf.pkl')
    pca = joblib.load('pca.pkl')

    X = (X - np.mean(X))/np.std(X, axis=0)
    Xt = pca.transform(X)
    return rbf.predict(Xt)





if __name__ == '__main__':
    Nwheel = 3
    D = pd.read_csv('data178586.csv', header = None)
    N = D.shape[0]
    delta = int(N/Nwheel)
    X = D.loc[:,:9]
    y = D.loc[:,10]

    print("variance in y %f" % np.std(y))
    for i in range(Nwheel):
        i1 = delta*i
        i2 = delta*(i+1)
        training = pd.concat([X[:i1],X[i2:]])
        test = X[i1:i2]
        # train
        ytrain = pd.concat([y[:i1],y[i2:]])
        train(training,ytrain)

        # test 
        yp = predict(test)
        print("average error: %0.3g" % error(yp, y[i1:i2]) )
        ypp = predict(training)
        print("training error: %0.3g" % error(ypp, ytrain) )