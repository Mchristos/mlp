# Machine Learning Fundamentals 
#### First-principles implementations of foundational algorithms

Simple, readable implementations of MLP (multi-layer perceptron) and RBF (radial basis function) networks. 

Interested in understanding the foundations of machine learning? Then implement the fundamental algorithms from first principles. Most implementations of machine learning algorithms are written for efficiency. It's hard to find clear, well-commented code that implements them. This repo aims for simplicity and readability, exposing how these algorithms can be implemented on a conceptual level.

## Multi-Layer Perceptron (MLP) 
An implementation of an MLP regressor (/classifier) implementing the sklearn api's like fit(), predict(), score() etc. Includes the following features: 

- Backpropogation 
- Mini-batch gradient descent 
- Momentum 
- Arbitrary number of hidden layers
- Random shuffling
- ReLU, Tanh, and Sigmoid activation functions.

### Example Usage (the XOR problem) 

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
        X = np.linspace(-0.5, 1.5, 100)
        Y = np.linspace(-0.5, 1.5, 100)
        X, Y = np.meshgrid(X, Y)
        def F(x,y):
            return mlp.predict(np.array([[x,y]]))
        Z = np.vectorize(F)(X,Y)
        plt.pcolor(X,Y,Z, cmap='RdBu')
        plt.colorbar()
        cntr = plt.contour(X,Y,Z, levels = [0.5])
        plt.clabel(cntr, inline=1, fontsize=10)
        plt.scatter([0,1], [0,1], s = 500, c = 'r')
        plt.scatter([1,0], [0,1], s = 500, marker = 'v')
        plt.grid()
        plt.show()

<img src="https://user-images.githubusercontent.com/13951953/47428181-608b4a00-d78a-11e8-9cc8-28fd0795e749.png" alt="drawing" width="500px"/>


## Radial Basis Function (RBF) 
A simple implementation for a simple algorithm. 

- Regularisation 
- Automatically compute centers using KMeans clustering

Inspired by scikit-learn: each class implements the sklearn api (e.g fit(), predict(), score() etc). 
