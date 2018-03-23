import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rbf import RBF
from mlp import MLP 
import json
from sklearn.externals import joblib
import time

def error(yp, y):
    """
    Compute error between predicted yp and actual y values.
    """
    # return 0.5*np.sum((yp - y)**2)
    return np.sqrt(sum((yp-y)**2)/y.shape[0])

def train_rbf(X, y):
    """ train rbf model, save model"""
    startTime = time.time()
    print("training...")
    # hyperparameters    
    n_pca_components = 10
    n_centers = 100
    # activation_sigma = 10

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
    rbf = RBF(centers, activation='linear')
    rbf.train(Xt, y) # dim = #centers 

    # save model 
    joblib.dump(rbf, 'rbf.pkl')
    joblib.dump(pca, 'pca.pkl')

    print("took %0.3f seconds" % (time.time() - startTime))

def predict_rbf(X):
    # load model 
    rbf = joblib.load('rbf.pkl')
    pca = joblib.load('pca.pkl')
    # predict 
    X = (X - np.mean(X))/np.std(X, axis=0)
    Xt = pca.transform(X)
    return rbf.predict(Xt)

def trainandtestwheel():
    Nwheel = 3
    D = pd.read_csv('data178586.csv', header = None)
    N = D.shape[0]
    delta = int(N/Nwheel)
    X = D.loc[:,:9]
    X = (X - X.mean())/X.std()
    y = D.loc[:,10]
    print("std in y %f" % np.std(y))
    for i in range(Nwheel):
        i1 = delta*i
        i2 = delta*(i+1)
        training = pd.concat([X[:i1],X[i2:]])
        test = X[i1:i2]
        # train
        ytrain = pd.concat([y[:i1],y[i2:]])
        # train_mlp(training,ytrain)


def dothethings():
    D = pd.read_csv('data178586.csv', header = None)
    X = D.loc[:,:9]
    X = (X - X.mean())/X.std()
    print(X)

    y = D.loc[:,10]
    ymean = y.mean()
    ystd = y.std()
    yt = (y - ymean)/ystd

    # hyperparameters 
    hiddenlayers = 5
    activation = 'sigmoid'
    n_pca_components = 5
    eta = 0.01
    iters = 2000
    # pre-process (PCA)
    pca = PCA(n_components = n_pca_components)
    pca.fit(X)
    Xt = pca.transform(X)
    # train
    T = yt.values.reshape(-1,1)
    mlp = MLP(Xt.shape[1],hiddenlayers, T.shape[1], eta = eta, activation=activation)
    mlp.train(Xt, T, epochs=iters)
    
    plt.plot(mlp.error)
    plt.show()


if __name__ == '__main__':
    dothethings()
