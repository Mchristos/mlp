import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from rbf import RBF
from mlp import MLP 
import json
from sklearn.externals import joblib
import time

def RMSE(yp, y):
    """
    Root Mean Squared Error 
    """
    return np.sqrt(sum((yp-y)**2)/y.shape[0])

def dotherbfthings():
    D = pd.read_csv('data178586.csv', header = None)
    X = D.loc[:,[2,4,5,6,9]]
    X = np.array((X - X.mean())/X.std())
    # X = X.reshape(-1,1)

    y = D.loc[:,10]
    ymean = y.mean()
    ystd = y.std()
    y = np.array((y - ymean)/ystd)

    # hyperparameters    
    n_pca_components = 1
    n_centers = 9
    activation = 'gaussian'
    activation_sigma = 0.1

    ## pca 
    # pca = PCA(n_components=n_pca_components)    
    # X = pca.fit_transform(X)
    # print("components: \n %r" % pca.components_) 
    # print("explained variance: \n %r " % pca.explained_variance_)
    # print("covariance \n %r" % pca.get_covariance())

    ## ica 
    # ica = FastICA(n_components=5)
    # X = ica.fit_transform(X)


    # cluster data to get centres 
    kmeans = KMeans(n_clusters = n_centers)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    print("centers = \n %r" % centers)

    # train RBF 
    rbf = RBF(centers, activation=activation, sigma=activation_sigma)
    rbf.train(X, y) # dim = #centers 
    error = RMSE(rbf.predict(X),y)
    print(error)


    # Xp = np.linspace(-3,3,1000).reshape(-1,1)
    # yp = rbf.predict(Xp)
    # plt.scatter(X, y)
    # plt.plot(Xp,yp, c='y')
    # plt.show()

    
def dothethings():
    D = pd.read_csv('data178586.csv', header = None)
    X = D.loc[:,:9]
    X = (X - X.mean())/X.std()

    y = D.loc[:,10]
    ymean = y.mean()
    ystd = y.std()
    yt = (y - ymean)/ystd
    mini = min(yt)
    maxi = max(yt)
    yt = (yt - mini)/(maxi - mini)

    # hyperparameters 
    hiddenlayers = [5]
    activation = 'sigmoid'
    n_pca_components = 3
    eta = 0.001
    iters = 1000
    # pre-process (PCA)
    # pca = PCA(n_components = n_pca_components)
    # pca.fit(X)
    # Xt = pca.transform(X)
    Xt = X
    # train
    T = yt.values.reshape(-1,1)
    mlp = MLP([Xt.shape[1],*hiddenlayers, T.shape[1]], eta = eta, activation=activation)
    startTime = time.time()
    mlp.train(Xt, T, epochs=iters)
    print("training time: %fs" % (time.time() - startTime) )
    yp = mlp.predict(Xt)
    plt.plot(mlp.error)
    plt.show()
    # reverse transform
    yp = (yp*(maxi-mini) + mini)*ystd + ymean

    print("training error: %f" % error(yp, np.array(y).reshape(-1,1))) 

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

if __name__ == '__main__':
    dotherbfthings()
