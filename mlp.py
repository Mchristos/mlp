import numpy as np

def SIG(x):
    """ sigmoid function """
    return 1/(1 + np.exp(-x))
def dSIG(x):
    """ derivative of sigmoid """
    return np.exp(-x)/(1 + np.exp(-x))**2

def addOnesCol(X):
    """ add column of ones """
    a, b = X.shape
    Xt = np.zeros([a, b + 1])
    Xt[:,1:] = X
    Xt[:,0] = np.ones(a)
    return Xt

class MLP():
    """
    Implements multi-layer perceptron 
    """
    def __init__(self, dim_in, dim_hidden, dim_out, eta, activation = 'sigmoid'):
        """
        eta = leraning rate 
        activation = activation function
        """
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.eta = eta
        if activation == 'sigmoid':
            self.f = SIG
            self.f_prime = dSIG
        if activation == 'linear':
            self.f = lambda x : x
            self.f_prime = lambda x : 1

    def train(self, X, T, iters, method = 'seq', V = None, W = None):
        X = addOnesCol(X)
        T = np.array(T)
        if X.shape[0] != T.shape[0]:
            raise ValueError("incorrect shape")
        # random initial weights
        if V is None:
            V = np.random.rand(self.dim_hidden, self.dim_in  + 1)
            W = np.random.rand(self.dim_out, self.dim_hidden + 1)
        # store error values 
        self.error = np.zeros(iters)
        # update weights 
        for t in range(iters):
            if method == 'batch':
                V, W, Y= self._batch_update(X, T, V, W)
                self.error[t] = self._error(Y, T)
            if method == 'seq':
                for i in range(X.shape[0]):
                    V, W = self._seq_update(X[i,:], T[i,:], V, W)
                self.error[t] = self._error2(X, V, W, T)
        self.V = V
        self.W = W
        return V, W

    def _batch_update(self, X, T, V, W):
        # forward pass 
        U = (V@X.T).T
        Z = addOnesCol(self.f(U))
        A = (W@Z.T).T
        Y = self.f(A)
        # backward pass
        Delta = -self.f_prime(A)*(T - Y)
        delta =  self.f_prime(U)*(Delta@W)[:,1:]
        #                                    ^ ignore zeroth component of hidden layer
        # for each forward/backward pass 
        for i in range(Delta.shape[0]):
            W_ = W - self.eta * (Delta[i,:].reshape(-1,1) @ Z[i,:].reshape(1,-1))
            V_ = V - self.eta * (delta[i,:].reshape(-1,1) @ X[i,:].reshape(1,-1))
        return V_, W_, Y
    
    def _owen_batch_update(self, X, T, V, W):
        # forward pass 
        U = (V@X.T).T
        Z = addOnesCol(self.f(U))
        A = (W@Z.T).T
        Y = self.f(A)
        # backward pass
        Delta = -self.f_prime(A)*(T - Y)
        delta = self.f_prime(U)*(Delta@W)[:,1:]
        #                                    ^ ignore zeroth component of hidden layer
        Delta = np.mean(Delta, axis = 0)
        delta = np.mean(delta, axis = 0)
        # for each forward/backward pass 
        # update weights 
        x = np.mean(X, axis = 0)
        z = np.mean(Z, axis = 0)
        V_ = V - self.eta * (delta.reshape(-1,1) @ x.reshape(1,-1))
        W_ = W - self.eta * (Delta.reshape(-1,1) @ z.reshape(1,-1))
        return V_, W_, Y


    def _seq_update(self, x, t, V, W):
        # forward pass
        u = V@x
        z = self.f(u)
        z = np.array([1.,*z]) # add one 
        a = W@z
        y = self.f(a)
        # backward pass 
        Delta = -self.f_prime(a)*(t - y) # row vector 
        delta =  self.f_prime(u)*(Delta@W)[1:]
        #                                  ^ ignore zeroth component 
        # update weights 
        V_ = V - self.eta * (delta.reshape(-1,1) @ x.reshape(1,-1))
        W_ = W - self.eta * (Delta.reshape(-1,1) @ z.reshape(1,-1))
        if(np.isnan(V_).any()):
            raise RuntimeError("nan values encountered")
        return V_, W_
    
    def predict(self, X):
        """ predict using trained model """
        # forward pass 
        X = addOnesCol(X)
        U = (self.V@X.T).T
        Z = addOnesCol(self.f(U))
        A = (self.W@Z.T).T
        Y = self.f(A)
        return Y 

    def _error(self, Y, T):
        """ compute error """
        return 0.5*np.sum((Y - T)**2)/len(Y)

    def _error2(self, X, V, W, T):
        """compute error given weights """
        U = (V@X.T).T
        Z = addOnesCol(self.f(U))
        A = (W@Z.T).T
        Y = self.f(A)
        return self._error(Y, T)

    def __repr__(self):
        return "in:%i, hidden:%i out:%i " % (self.dim_in, self.dim_hidden, self.dim_out)