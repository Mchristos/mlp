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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def RMSE(yp, y):
    """
    Root Mean Squared Error 
    """
    return np.sqrt(sum((yp-y)**2)/y.shape[0])

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

def rbf_train(D):
    X = np.array(D.loc[:,[4,6,8]])
    y = np.array(D.loc[:,10]) 
    Xmeans = np.mean(X, axis = 0)
    Xstds  = np.std(X, axis = 0)
    X = (X - Xmeans)/Xstds
    ymean = y.mean()
    ystd = y.std()
    y = (y - ymean)/ystd
    # hyperparameters    
    n_centers = 20 # 900
    activation = 'gaussian'
    activation_sigma = 0.5
    # train RBF 
    rbf = RBF(n_centers, activation=activation, sigma=activation_sigma)
    rbf.fit(X, y) # dim = #centers 
    # save model
    model = {
        'Xmeans' : Xmeans.tolist(),
        'Xstds'  : Xstds.tolist(),
        'ymean'  : ymean,
        'ystd'   : ystd
    }
    with open('model.json', 'w') as f:
        f.write(json.dumps(model))
    joblib.dump(rbf, 'rbf.pkl')
    error = RMSE(rbf.predict(X),y)
    print("training error: %f" % error )

def rbf_test(D):
    with open('model.json', 'r') as f:
        model = json.loads(f.read())
    rbf = joblib.load('rbf.pkl')
    X = np.array(D.loc[:,[4,6,8]])
    y = np.array(D.loc[:,10])
    # transform data
    X = (X - model['Xmeans'])/model['Xstds']
    y = (y - model['ymean'])/model['ystd']
    # predict 
    error = rbf.score(X,y)
    print("test error:     %f" % error )

def trainandtestwheel():
    D = pd.read_csv('data178586.csv', header = None)
    Nwheel = 3
    N = D.shape[0]
    delta = int(N/Nwheel)
    randshift = np.random.randint(0,N)
    for i in range(Nwheel):
        i1 = (delta*i + randshift) % N 
        i2 = (delta*(i+1) + randshift) % N 
        Dtrain = pd.concat([D[:i1],D[i2:]])
        Dtest = D[i1:i2]
        rbf_train(Dtrain)
        rbf_test(Dtest)
        print("")

def grid_search():
    D = pd.read_csv('data178586.csv', header = None)
    X = np.array(D.loc[:,[4,6,8]])
    y = np.array(D.loc[:,10]) 
    Xmeans = np.mean(X, axis = 0)
    Xstds  = np.std(X, axis = 0)
    X = (X - Xmeans)/Xstds
    ymean = y.mean()
    ystd = y.std()
    y = (y - ymean)/ystd
    rbf = RBF(n_centers = -1, activation='gaussian', sigma=-1)
    c_range = [int(x) for x in np.linspace(800,1000,50)]
    sig_range = np.linspace(0.4,0.6,50)
    param_grid = dict(n_centers = c_range, sigma = sig_range)
    def scoring_func(estimator, X, y):
        return estimator.error(X,y)
    grid = RandomizedSearchCV(rbf, param_grid, n_iter=300, cv=3, n_jobs=-1, 
                                               scoring=scoring_func, refit=False)
    grid.fit(X,y)
    # save results 
    scores = grid.grid_scores_
    with open('cv_results.json','w') as f:
        f.write(json.dumps(grid.cv_results_, cls=NumpyEncoder))
    with open('cv_results.txt','w') as f:
        for tpl in scores:
             f.write(str(tpl) + "\n")


if __name__ == '__main__':

    with open("cv_results.json",'r') as f:
        cv_results = json.load(f)
    
    n_centers = np.array(cv_results['param_n_centers'])
    sigma = np.array(cv_results['param_sigma'])
    scores = np.array(cv_results['mean_test_score'])
    x,y = np.meshgrid(n_centers, sigma)
    plt.pcolor(x,y, scores)
    # plt.scatter(n_centers, sigma)
    plt.show()
    # D = pd.read_csv('data178586.csv', header = None)
    # Dtrain, Dtest = train_test_split(D, test_size = 0.333333333333333333333)
    # rbf_train(Dtrain)
    # rbf_test(Dtest)
