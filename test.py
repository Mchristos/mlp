import unittest
import numpy as np 
from mlp import MLP 

threshold = 0.0000001 # small value for testing equality 

class TestMLP(unittest.TestCase):

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

    # def test_upper(self):
    #     self.assertEqual('foo'.upper(), 'FOO')

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())

    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    unittest.main()     