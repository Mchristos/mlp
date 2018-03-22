import unittest
import numpy as np 
from mlp import MLP 
import time 

threshold = 0.0000001 # small value for testing equality 

class TestMLP(unittest.TestCase):
    
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f" % (self.id(), t))

    def test_basic(self):
        mlp = MLP(2, 2, 2, eta = 0.1, activation='linear')
        X = np.array([[0,1.]])
        T = np.array([[1.,0]])
        V = np.array([[1,-2, 1], 
                    [1, 0, 1]])
        W = np.array([[1, 1, 1],
                    [1,-1, 1]])
        mlp.train(X, T, 1, V = V, W = W)
        actual = mlp.predict(X)
        expected = np.array([[1.08, 0.02]])
        delta = actual - expected
        self.assertTrue(np.all(delta < threshold))

    def test_XOR(self):
        mlp = MLP(2, 5, 1, eta = 0.1, activation = 'sigmoid')
        X = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
        T = np.array([[0],
                      [1],
                      [1],
                      [0]])
        mlp.train(X, T, 9000, method = 'seq')
        prediction = mlp.predict(X)
        self.assertTrue(np.all( (prediction > 0.5) == T))
    
    def test_linear(self):
        mlp = MLP(1, 5, 1, eta = 0.01, activation='linear')
        X = np.linspace(-1,1,10).reshape(-1,1)
        T = np.linspace(6,9,10).reshape(-1,1)
        mlp.train(X, T, 1000)
        Tp = mlp.predict(X)
        self.assertTrue( np.all((Tp - T)<threshold) )
    
    def test_sin(self):
        mlp = MLP(1, 5, 1, eta = 0.01, activation='sigmoid')
        n = 100
        X = np.random.rand(n).reshape(-1,1)
        T = np.sin(np.pi*X) #+ np.random.normal(size = n, scale =  0.05).reshape(-1,1)
        mlp.train(X, T, 1000)
        Xp = np.random.rand(n).reshape(-1,1)
        Tp = mlp.predict(Xp)
        self.assertTrue(False)




if __name__ == '__main__':
    unittest.main()     