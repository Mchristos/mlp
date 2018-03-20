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
        self.f = SIG
        self.g = SIG
        self.g_prime = dSIG
 
    
    def train_batch(self, X, T, iters):
        X = np.array(X)
        T = np.array(T)
        if X.shape[0] != T.shape[0]:
            raise ValueError("incorrect shape")
                # random initial weights 
        V = np.random.rand(self.dim_hidden, self.dim_in)
        W = np.random.rand(self.dim_out, self.dim_hidden)
        for t in range(iters):
            V, W = self._batch_update(X, T, V, W)
        self.V = V
        self.W = W
        return V, W

    def _batch_update(self, X, T, V, W):
        # forward pass 
        U = (V@X.T).T
        Z = self.f(U)
        A = (W@Z.T).T
        Y = self.f(A)
        # print("shapes: U:%r, Z:%r, A:%r, Y:%r" % (U.shape,Z.shape,A.shape,Y.shape))
        # backward pass
        Delta = -self.g_prime(A)*(T - Y)
        delta = self.g_prime(U)*(Delta@W)
        for i in range(Delta.shape[0]):
            V = V - self.eta * (delta[i,:].reshape(-1,1) @ X[i,:].reshape(1,-1))
            W = W - self.eta * (Delta[i,:].reshape(-1,1) @ Z[i,:].reshape(1,-1))
        return V, W

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
        Y = self.g(self.W@self.f(self.V@X.T)).T
        return Y 

    def _sequential_update(self, x, t, V, W):
        # forward pass
        u = V@x
        z = self.f(u)
        a = W@z
        y = self.g(a)
        # backward pass 
        Delta = -self.g_prime(a)*(t - y) # row vector 
        delta = self.g_prime(u)*(Delta@W)
        # update weights 
        V_ = V - self.eta * (delta.reshape(-1,1) @ x.reshape(1,-1))
        W_ = W - self.eta * (Delta.reshape(-1,1) @ z.reshape(1,-1))
        if(np.isnan(V_).any()) :
            raise RuntimeError("nan values encountered")
        return V_, W_

    def _error(self, X, T, V, W):
        Y = self.g(W@self.f(V@X.T)).T
        return 0.5*np.sum((Y - T)**2)
    
    def __repr__(self):
        return "in:%i, hidden:%i out:%i " % (self.dim_in, self.dim_hidden, self.dim_out)
