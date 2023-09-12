# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 14:37:13 2023

@author: mytzi
"""
 #
#aqui se cargaron los archivos anteriores donde se definieron la red neuronal y 
#el procesamiento de los datos de mnist
import red_neuronal_mnist
import mnist

#Realizamos la segmentacion de los datos usando la funcion load_data_wrapper() 
#definida en el archivo de mnist
training_data, validation_data , test_data = mnist.load_data_wrapper() 

#Convertimos estos datos en el tipo lista
training_data = list(training_data)
test_data = list(test_data)

#Definimos una red neuronal de 784 neuronas de entrada, 30 intermedias y 10 de salida
#esto es, una red neuronal con 3 capas
net = red_neuronal_mnist.Network([784,30,10])


#Entrenamos la red neuronal con el optimizador SGD usando 10 epocas, 2 mini batches 
#y un learning rate de 3, al final evaluamos el modelo con test_data
net.SGD_momentum( training_data, 10, 2, 3.0, .82, test_data=test_data)


#Mejoro un 1% la predicción, pues en la version anteriorior tenia un 92% de prediccion 
#Ahora tiene el 93%