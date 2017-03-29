'''
Created on Mar 28, 2017

@author: Leo Zhong
'''
import numpy as np # for metrics calculation

#define functions
def tanh(x):  
    return np.tanh(x)

def tanh_deriv(x):  #derivative for tanh
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):  
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):  #derivative for logistic function
    return logistic(x)*(1-logistic(x))



class NeuralNetwork:   #object oriented
    # __init__ : constructor
    # self = this in java
    # layers: # of unit in each level
    # activation: choose function => default is tanh
    
    def __init__(self, layers, activation='tanh'):  
        
        # set function
        if activation == 'logistic':  
            self.activation = logistic  
            self.activation_deriv = logistic_derivative  
        elif activation == 'tanh':  
            self.activation = tanh  
            self.activation_deriv = tanh_deriv
    
        
        self.weights = []  # store weight
        #initialize weight
        for i in range(1, len(layers) - 1):  
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)  
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)
            
            
    def fit(self, X, y, learning_rate=0.2, epochs=10000):  
        # X: object y: class lable  learning_rate: 0 ~ 1     
        X = np.atleast_2d(X)    # ensure at least 2D  
        temp = np.ones([X.shape[0], X.shape[1]+1])  # initialize metrics       
        # X.shape[0]: row, X.shape[1]: col
        temp[:, 0:-1] = X  # adding the bias unit to the input layer         
        X = temp         
        y = np.array(y) # to np array
    
        for k in range(epochs):  
            i = np.random.randint(X.shape[0])  
            a = [X[i]]
    
            for l in range(len(self.weights)):  #going forward network, for each layer
                a.append(self.activation(np.dot(a[l], self.weights[l])))  #Computer the node value for each layer (O_i) using activation function
            error = y[i] - a[-1]  #Computer the error at the top layer
            deltas = [error * self.activation_deriv(a[-1])] #For output layer, Err calculation (delta is updated error)
            
            #Staring backprobagation
            for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer 
                #Compute the updated error (i,e, deltas) for each node going from top layer to input layer 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))  
            deltas.reverse()  
            for i in range(len(self.weights)):  
                layer = np.atleast_2d(a[i])  
                delta = np.atleast_2d(deltas[i])  
                self.weights[i] += learning_rate * layer.T.dot(delta)
                
                
    def predict(self, x):         
        x = np.array(x)         
        temp = np.ones(x.shape[0]+1)         
        temp[0:-1] = x         
        a = temp         
        for l in range(0, len(self.weights)):             
            a = self.activation(np.dot(a, self.weights[l]))         
        return a


