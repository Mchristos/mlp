import unittest
import numpy as np 
from mlp import MLP
from rbf import RBF
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
import time 
import matplotlib.pyplot as plt 


threshold = 0.0001 # small value for testing equality 
def RMSE(Y, T):
    """ Root Mean Squared Error """
    return np.sqrt(np.sum((Y - T)**2)/len(Y))

class TestRBF(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f" % (self.id(), t))
    
    def test_sin(self):
        n = 10000
        X = np.random.rand(n).reshape(-1,1)
        noise = 0.3
        T = 0.5*np.sin(4*np.pi*X) + 0.5 + np.random.normal(size = n, scale = noise).reshape(-1,1)
        rbf = RBF(n_centers=20, activation='gaussian', sigma = 0.05)
        rbf.fit(X,T)
        Tp = rbf.predict(X)
        error = RMSE(Tp, T)
        # Xp = np.linspace(0,1,1000).reshape(-1,1)
        # Tp = rbf.predict(Xp)
        # plt.scatter(X,T)
        # plt.plot(Xp,Tp, c = 'y')
        # plt.show()
        epsilon = 0.005
        self.assertTrue(error < noise + epsilon)

    def test_reg(self):
        # sinusoidal function 
        def f(x):
            return 0.5*np.sin(4*np.pi*x) + 0.5
        # train on data noisily following f 
        n = 80
        X = np.random.rand(n).reshape(-1,1)
        noise = 0.05
        T = f(X) + np.random.normal(size = n, scale = noise).reshape(-1,1)
        rbf = RBF(n_centers=20, activation='gaussian', sigma = 0.05, lambdaReg=20.)
        rbf.fit(X,T)
        xl = np.linspace(0,1,1000).reshape(-1,1)
        yl = rbf.predict(xl)
        # plt.scatter(X, T)    # training data 
        # plt.plot(xl, f(xl))  # true curve 
        # plt.plot(xl,yl)      # learned curve 
        # plt.show()
        epsilon = 0.01
        true_error = RMSE(yl, f(xl))  
        self.assertLess(true_error, noise + epsilon)

    def test_sin_redundancy(self):
        n = 1000
        X1 = np.random.rand(n).reshape(-1,1)
        X2 = np.random.rand(n).reshape(-1,1) # redundant dimension
        X = np.concatenate([X1, X2], axis = 1)
        noise = 0.05
        T = 0.5*np.sin(4*np.pi*X1) + 0.5 + np.random.normal(size = n, scale = noise).reshape(-1,1)
        # rbf train 
        rbf = RBF(n_centers=150, activation='gaussian', sigma = 0.3, lambdaReg=1e-6)
        rbf.fit(X,T)
        # predict
        Tp = rbf.predict(X)
        error = RMSE(Tp, T)
        # Xp1 = np.linspace(0,1,1000).reshape(-1,1)
        # Xp2 = np.random.rand(1000).reshape(-1,1) # random 2nd co-ordinate 
        # Xp = np.concatenate([Xp1,Xp2], axis = 1)
        # Tp = rbf.predict(Xp)
        # plt.scatter(X1,T)
        # plt.plot(Xp1.reshape(-1,1) ,Tp, c = 'y')
        # plt.show()
        epsilon = 0.01
        self.assertTrue(error < noise + epsilon)
    
    def test_XOR(self):

        X = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
        T = np.array([[0],
                      [1],
                      [1],
                      [0]])
        rbf = RBF(centers=X) # centers are data itself 
        rbf.fit(X, T)
        prediction = rbf.predict(X)
        self.assertTrue(np.all( (prediction > 0.5) == T))


class TestMLP(unittest.TestCase):
    
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f" % (self.id(), t))

    def test_basic(self):
        mlp = MLP(dims=[2,2,2], eta = 0.1, activation='linear', max_epochs=1)
        X = np.array([[0,1.]])
        T = np.array([[1.,0]])
        V = np.array([[1,-2, 1], 
                    [1, 0, 1]])
        W = np.array([[1, 1, 1],
                    [1,-1, 1]])
        mlp.fit(X, T, weights = [V,W])
        actual = mlp.predict(X)
        expected = np.array([[1.08, 0.02]])
        delta = actual - expected
        self.assertTrue(np.all(delta < threshold))

    def test_XOR(self):
        mlp = MLP(dims =[2, 5, 1], eta = 0.1, activation = 'sigmoid', max_epochs=4000, alpha=0.55)
        X = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
        T = np.array([[0],
                      [1],
                      [1],
                      [0]])
        mlp.fit(X, T)
        prediction = mlp.predict(X)
        self.assertTrue(np.all( (prediction > 0.5) == T))
    
    def test_linear_overfit(self):
        mlp = MLP([1, 5, 1], eta = 0.01, activation='linear', max_epochs=1000, alpha=0.01)
        X = np.linspace(-1,1,10).reshape(-1,1)
        T = np.linspace(6,7,10).reshape(-1,1)
        mlp.fit(X, T)
        Tp = mlp.predict(X)
        # plt.figure(1)
        # plt.plot(mlp.error)
        # plt.figure(2)
        # plt.scatter(X,T)
        # plt.plot(X,Tp, c = 'y') 
        # plt.show()
        self.assertTrue(np.allclose(T, Tp))
    
    def test_sin(self):
        mlp = MLP([1, 4,2, 1], eta = 0.01, activation='sigmoid', stochastic = 0.8, max_epochs=5000, alpha=0.9)
        n = 1000
        X = np.random.rand(n).reshape(-1,1)
        noise = 0.05
        T = 0.5 + 0.3*np.sin(2*np.pi*X) + np.random.normal(size = n, scale = noise).reshape(-1,1)
        mlp.fit(X, T)
        Xp = np.linspace(0,1,1000).reshape(-1,1)
        Yp = mlp.predict(Xp)
        # plt.figure(1)
        # plt.plot(mlp.error)
        # plt.figure(2)
        # plt.scatter(X,T)
        # plt.plot(Xp,Yp, c = 'y') 
        # plt.show()
        self.assertLess(mlp.error[-1], noise + 0.1*noise)

if __name__ == '__main__':
    # test = TestMLP()
    # test.test_sin()
    unittest.main()     