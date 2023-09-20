# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 01:54:31 2023   

@author: mytzi
"""
"""
IMPORTANTE
Nota primera tarea: 
Este código lo comenté desde que comencé a correrlo para entender 
lo que hacia, entonces, cuando entrene la red ya habia documentado el 
código, por esto mismo, no hay modificaciones nuevas.
9 
"""
#Mejoro un 1% en predicción 
#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object): #clase conjunto de funciones

    def __init__(self, sizes): #self es el objeto y el sizes es el numero de neuronas
        self.num_layers = len(sizes) #aqui se define el numero de capas
        self.sizes = sizes #aqui se define el tamaño de la red
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #se inicia los umbrales aleatoriamente y los pesos
        self.weights = [np.random.randn(y, x) #pesos, llega hasta penultima capa porquela ultima ya no tiene peso pero si umbrales
                        for x, y in zip(sizes[:-1], sizes[1:])] 
        
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = sigmoid(z) if w != self.weights[-1] else self.softmax(z) 
        return a.any()
    def softmax(self, z): 
        exp_z = np.exp(z - np.max(z))  # Subtracting the max value for numerical stability 
        return exp_z / exp_z.sum(axis=0, keepdims=True)
    def cost_derivative(self, output_activations, y):
        return output_activations - y  # This assumes y is a one-hot encoded vector

# En este caso, la función softmax se aplicó en la capa de salida, por lo que output_activations representa las probabilidades.
    #Definimos una nueva función para el optimizador SGD con inercia 
    def SGD_momentum(self, training_data, epochs, mini_batch_size, eta, momentum,
            test_data=None):

        training_data = list(training_data)
        n = len(training_data)
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        for j in range(epochs): #segmentacion de acuerdo al numero de epocas que tenemos. 
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)] #segmentacion de los datos de entrenamiento, 
            #los convertimos en mini-batches 
            for mini_batch in mini_batches:
                self.update_mini_batch_SGD_momentum(mini_batch, eta, momentum)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
                #genera el numero de epocas  y el numero total de datos de prueba
            else:
                print("Epoch {} complete".format(j))
                
    def update_mini_batch_SGD_momentum(self, mini_batch, eta, momentum ):
        #vector de ceros para vdw y vdb, los cuales 
        #almacenaran informacion sobre el momento 
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]#vector de ceros
        vdw=[np.zeros(w.shape) for w in self.weights] 
        vdb=[np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] #Actualizamos la parcial de w 
            vdw = [momentum*nvdw + (1-momentum)*(dnw) for nvdw, dnw in zip(vdw, nabla_w)]
            vdb = [momentum*nvdb +(1-momentum)*(dnb) for nvdb,dnb in zip(vdb, nabla_b)]

        self.weights = [w-eta*nvdw for w, nvdw in zip(self.weights, vdw)]
        self.biases = [b-eta*nvdb for b, nvdb in zip(self.biases, vdb)]


    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):#esta funcion llamara a otras funciones, es el optimizador sgd tuples no se pueden editar
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
        
        #Por aqui iria el softmax
        # Actualizamos el backprop (porque cambiamos la funcion de costo)
        
        delta = 1/len(activations)*self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1]) #error de la ultima capa
        nabla_b[-1] = delta  #derivada parcial de la funcion de costo respecto a b
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) 
        #Derivada parcial de la funcion de costos respecto a w
        
        
        for l in range(2, self.num_layers): #actualizacion por cada capa comenzando de la ultima hasta la segunda
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta #errores de b y de w, es mas como una variación
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data): #datos de prueba
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

     #Actualizamos la derivada de la funcion de costos 
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)/output_activations*(1-output_activations)

#### Miscellaneous functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

"""Aqui intente definir la funcón softmax, pero no pude entrelazar con lo que me 
arrojaba el codigo, para que tomara la ultima capa"""
#def softmax(z,zs)
#    return np.exp(z)/zs
