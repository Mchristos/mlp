import numpy as np
import matplotlib.pyplot as plt 

def SIG(x):
    """ sigmoid function """
    return 1/(1 + np.exp(-x))
def dSIG(x):
    """ derivative of sigmoid """
    return np.exp(-x)/(1 + np.exp(-x))**2

class MLP():
    """
    Implements multi-layer perceptron 
    """
    def __init__(self, dim_in, dim_hidden, dim_out, eta):
        """
        eta = leraning rate 
        """
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.eta = eta
        self.V = np.zeros([self.dim_hidden, self.dim_in])
        self.W = np.zeros([self.dim_out, self.dim_hidden])

    def train_batch(self, X, T):
        Y = SIG(self.W@SIG(self.V@X.T)).T

        return None

    def train_seq(self, X, T, iters, ploterror = False):
        X = np.array(X)
        T = np.array(T)
        if X.shape[0] != T.shape[0]:
            raise ValueError("incorrect shape")
        if ploterror:
            err = np.zeros(iters)
        # initialize weights 
        V = np.random.rand(self.dim_hidden, self.dim_in)
        W = np.random.rand(self.dim_out, self.dim_hidden)
        # update sequentially with backprop 
        for t in range(iters):
            for i in range(X.shape[0]):
                V, W = self._sequential_update(X[i,:], T[i,:], V, W)
            if ploterror:
                err[t] = self._error(X, T, V, W)
        # plot error 
        if ploterror:
            print(err)
            plt.plot(err)
            plt.show()
        # set learned weights 
        self.V = V
        self.W = W
        return V, W

    def predict(self, X):
        Y = SIG(self.W@SIG(self.V@X.T)).T
        return Y 

    def _sequential_update(self, x, t, V, W):
        # forward pass
        u = V@x
        z = SIG(u)
        a = W@z
        y = SIG(a)
        # backward pass 
        Delta = -dSIG(a)*(t - y) # row vector 
        delta = dSIG(u)*(Delta@W)
        # update weights 
        V_ = V - self.eta * (delta.reshape(-1,1) @ x.reshape(1,-1))
        W_ = W - self.eta * (Delta.reshape(-1,1) @ z.reshape(1,-1))
        if(np.isnan(V_).any()) :
            raise RuntimeError("nan values encountered")
        return V_, W_

    def _error(self, X, T, V, W):
        Y = SIG(W@SIG(V@X.T)).T
        return 0.5*np.sum((Y - T)**2)
    
    def __repr__(self):
        return "in:%i, hidden:%i out:%i " % (self.dim_in, self.dim_hidden, self.dim_out)
