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
    def __init__(self, dims, eta, activation = 'sigmoid', max_epochs = 10000, deltaE = 1e-12):
        """
        dims = [dim_in, dim_hidden1, ..., dim_hiddenN, dim_out]
        eta = leraning rate 
        activation = activation function
        """
        self.set_params(dims,eta,activation,max_epochs, deltaE)

    # compatibility with sklearn 
    def set_params(self, dims, eta, activation = 'sigmoid', max_epochs = 10000, deltaE = 1e-12):
        self.dims = dims 
        self.eta = eta
        self.max_epochs = max_epochs
        self.activation = activation
        self.deltaE = deltaE
        if activation == 'sigmoid':
            self.f = SIG
            self.f_prime = dSIG
        if activation == 'linear':
            self.f = lambda x : x
            self.f_prime = lambda x : 1
        return self

    def get_params(self, deep = False):
        result = {
            'dims'    : self.dims,
            'eta'     : self.eta,
            'max_epochs'  : self.max_epochs,
            'activation' : self.activation
        }
        return result

    def fit(self, X, y, computeError = False, weights = None):
        if X.shape[0] != y.shape[0]:
            raise ValueError("incorrect shape")
        # random initial weights
        if weights is None:
            weights = []
            for i in range(len(self.dims)-1):
                W = np.random.rand(self.dims[i+1], self.dims[i] +1)
                #                  ^ output dim    ^ input dim plus bias dim 
                weights.append(W)
        # store error values 
        self.error = np.zeros(self.max_epochs+1)
        self.error[-1] = np.infty
        # main training loop
        t = 0 
        while (t < self.max_epochs):
            # forward pass 
            Y, x, u = self._forwardpass(X, weights)
            # compute error 
            error = self._RMSE(Y, y)
            self.error[t] = error
            delta = self.error[t-1] - self.error[t] 
            if delta < self.deltaE:
                break
            else:
            # backward pass 
                weights = self._backwardpass(y, Y, x, u, weights)
                t = t + 1
        self.error = self.error[:t]
        self.weights = weights
        return weights
   
    def predict(self, X):
        Y, _, __ = self._forwardpass(X, self.weights)
        return Y 
    
    def score(self, X, y):
        yp = self.predict(X)
        return self._RMSE(y,yp)

    def _RMSE(self,y, yp):
        """
        Computes the root mean squared error (RMSE)
        """
        return np.sqrt(np.sum((yp-y)**2)/y.shape[0])

    def _forwardpass(self, X, weights):
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

    def _backwardpass(self, y, Y, x, u, weights):
        """ 
        Compute updated weights by doing backward pass
        y = target 
        Y = true output 
        x = inputs to weight matrices at each layer during forward pass
        u = activations at each output layer during forward pass 
        """
        # backward pass
        D = -self.f_prime(u[-1])*(y - Y) # Delta 
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
        return weights_new
        
    def __repr__(self):
        return "in:%i, hidden:%i out:%i " % self.dims
    
    

