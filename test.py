import unittest
import numpy as np 
from mlp import MLP 
import time 
import matplotlib.pyplot as plt 


threshold = 0.0000001 # small value for testing equality 

class TestMLP(unittest.TestCase):
    
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f" % (self.id(), t))

    def test_basic(self):
        mlp = MLP(dims=[2,2,2], eta = 0.1, activation='linear')
        X = np.array([[0,1.]])
        T = np.array([[1.,0]])
        V = np.array([[1,-2, 1], 
                    [1, 0, 1]])
        W = np.array([[1, 1, 1],
                    [1,-1, 1]])
        mlp.train(X, T, 1, method = 'batch', weights = [V,W])
        actual = mlp.predict(X)
        expected = np.array([[1.08, 0.02]])
        delta = actual - expected
        self.assertTrue(np.all(delta < threshold))

    def test_XOR(self):
        mlp = MLP(dims =[2, 5, 1], eta = 0.1, activation = 'sigmoid')
        X = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
        T = np.array([[0],
                      [1],
                      [1],
                      [0]])
        mlp.train(X, T, 9000, method = 'batch')
        prediction = mlp.predict(X)
        self.assertTrue(np.all( (prediction > 0.5) == T))
    
    def test_linear(self):
        mlp = MLP([1, 5, 1], eta = 0.01, activation='linear')
        X = np.linspace(-1,1,10).reshape(-1,1)
        T = np.linspace(6,7,10).reshape(-1,1)
        mlp.train(X, T, 1000, method='batch')        
        Tp = mlp.predict(X)
        self.assertTrue( np.all((Tp - T)<threshold) )
    
    def test_sin(self):
        mlp = MLP([1, 5, 1], eta = 0.1, activation='sigmoid')
        n = 100
        X = np.random.rand(n).reshape(-1,1)
        T = 0.5*np.sin(2*np.pi*X) + 0.5 + np.random.normal(size = n, scale =  0.05).reshape(-1,1)
        mlp.train(X, T, 10000, method='batch')
        # Xp = np.linspace(0,1,1000).reshape(-1,1)
        # Yp = mlp.predict(Xp)
        # plt.plot(Xp,Yp)
        # plt.scatter(X,T)
        # plt.show()
        self.assertTrue(mlp.error[-1] < 0.02)

if __name__ == '__main__':
    unittest.main()     