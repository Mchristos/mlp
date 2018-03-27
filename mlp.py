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
    def __init__(self, dims, eta, activation = 'sigmoid'):
        """
        dims = [dim_in, dim_hidden1, ..., dim_hiddenN, dim_out]
        eta = leraning rate 
        activation = activation function
        """
        self.dims = dims 
        self.eta = eta
        if activation == 'sigmoid':
            self.f = SIG
            self.f_prime = dSIG
        if activation == 'linear':
            self.f = lambda x : x
            self.f_prime = lambda x : 1

    def train(self, X, T, epochs, method = 'batch', weights = None):
        T = np.array(T)
        if X.shape[0] != T.shape[0]:
            raise ValueError("incorrect shape")
        # random initial weights

        if weights is None:
            weights = []
            for i in range(len(self.dims)-1):
                w = np.random.rand(self.dims[i+1], self.dims[i] +1)
                #                  ^ output dim    ^ input dim plus bias dim 
                weights.append(w)
        # store error values 
        self.error = np.zeros(epochs)
        # update weights 
        for t in range(epochs):
            if method == 'batch':
                weights, Y= self._batch_update(X, T, weights)
                self.error[t] = self._error(Y, T)
            # if method == 'seq':
                # for i in range(X.shape[0]):
                #     V, W = self._seq_update(X[i,:], T[i,:], V, W)
                # self.error[t] = self._error2(X, V, W, T)
        self.weights = weights
        return weights
   
    def _batch_update(self, X, T, weights):
        # forward pass
        Y, x, u = self.forwardpass(X, weights)           

        # backward pass
        D = -self.f_prime(u[-1])*(T - Y) # Delta 
        delta = [D]
        for i in range(len(weights) -1):
            W = weights[::-1][i]   # go through weight matrices in reverse
            U = u[::-1][i+1] # go through outputs in reverse, from second last
            d = self.f_prime(U)*(delta[i]@W)[:,1:]
            delta.append(d)
        delta.reverse() # reverse delta!  

        # update weights 
        weights_new = []
        for i in range(len(weights)):
            temp = self.eta*(delta[i].T @ x[i])
            weights_new.append(weights[i] - temp)

        return weights_new, Y

    def forwardpass(self, X, weights):
        """ perform forward pass, saving values"""
        Y = X 
        x = [] # inputs to next layer 
        u = [] # activations  
        for i in range(len(weights)):
            X = addOnesCol(Y)
            x.append(X)             # save input 
            U = (weights[i]@X.T).T  # apply weight matrix
            u.append(U)             # save output
            Y = self.f(U)           # activated output  
        return Y , x, u

    def predict(self, X):
        Y, _, __ = self.forwardpass(X, self.weights)
        return Y 

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
    

    def _error(self, Y, T):
        """ compute error """
        return 0.5*np.sum((Y - T)**2)/len(Y)
        
    def __repr__(self):
        return "in:%i, hidden:%i out:%i " % self.dims