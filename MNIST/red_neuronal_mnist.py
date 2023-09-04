# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 01:54:31 2023

@author: mytzi
"""
"""
IMPORTANTE
Este código lo comente desde que comence a correrlo para entender 
lo que hacia, por lo cual, cuando entrene la red ya habia documentado el 
código, por esto mismo, no hay modificaciones nuevas. 
"""
# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object): #clase conjunto de funciones

    def __init__(self, sizes): #self es el objeto y el sizes es el numero de neuronas
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3 , 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes) #aqui se define el numero de capas
        self.sizes = sizes #aqui se define el tamaño de la red
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #se inicia los umbrales aleatoriamente y los pesos
        self.weights = [np.random.randn(y, x) #pesos, llega hasta penultima capa porquela ultima ya no tiene peso pero si umbrales
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a): #self es la red y a es el dato de entrada
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b) #ahora a es la funcion simoide, ie se actualiza b=bias
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):#esta funcion llamara a otras funciones, es el optimizador sgd tuples no se pueden editar
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data) #define datos de entrenamiento
        n = len(training_data) #cuenta cuantos datos de entrenamiento hay en el conjunto

        if test_data: #aqui hace lo mismo con los datos de prueba si es que hay
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs): #
            random.shuffle(training_data) #desordena los datos para tomar una muestra aleatoria
            mini_batches = [
                training_data[k:k+mini_batch_size] #agrupa los datos de entrenamiento en batches
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta) #actualizacion de w y b en cada batch
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta): #eta es un factor, lo debemos encontrar
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #hacemos un vector de ceros de bias y pesos para actualizarlo
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #entrenamiento y podemos ver que tanto varia la funcion de costo
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #actualisamos las parciales de los pesos y bias, notemos que sumamos entrada por entrada
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw #aqui actualizamos "optimizador" cada w y b se actualiza con su parcial correspondiente
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #creamos vectores cero con la longuitud b
        nabla_w = [np.zeros(w.shape) for w in self.weights] #creamos vectores cero con la longuitud w
        # feedforward
        activation = x #definimos un activador (dato de entrada)
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z) #vector con todos los valores de z
            activation = sigmoid(z) #activacion ya redefinida con la funcion sigmoide
            activations.append(activation) #se guarda en la lista
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1]) #esto es el error
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers): #actualizacion por cada capa comenzando de la ultima hasta la segunda
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta #errores de b y de w, es mas como una variación
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data): #datos de prueba
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


