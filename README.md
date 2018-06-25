# basic-ml

Simple, readable implementations of MLP (multi-layer perceptron) and RBF (radial basis function) networks. 

Most implementations of fundamental machine learning algorithms are written for efficiency. It's hard to find clear, well-commented code that implements them. This repo aims for simplicity and readability, exposing how these algorithms can be implemented on a conceptual level 

## Multi-Layer Perceptron (MLP) 
An implementation of an MLP regressor (/classifier) implementing the sklearn api's like fit(), predict(), score() etc. Includes the following features: 

- Backpropogation 
- Mini-batch gradient descent 
- Momentum 
- Arbitrary number of hidden layers
- Random shuffling
- ReLU, Tanh, and Sigmoid activation functions.  

## Radial Basis Function (RBF) 
A simple implementation for a simple algorithm. 

- Regularisation 
- Automatically compute centers using KMeans clustering

Inspired by scikit-learn: each class implements the sklearn api (e.g fit(), predict(), score() etc). 
