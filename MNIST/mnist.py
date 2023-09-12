# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 14:36:25 2023

@author: mytzi
"""

import pickle
import gzip

import numpy as np
 
#
#Se define una funcion que carga los datos de mnist y los segmenta en training, validation y test
def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

#Esta funcion procesa los datos de mnist 
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    #se redimensionan los datos de entrenamiento
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    #se vectorizan los resultados de entrenamiento 
    training_results = [vectorized_result(y) for y in tr_d[1]]
    #Unimos los datos de entrenamiento con sus resultados ya reordenados
    training_data = zip(training_inputs, training_results)
    #se realiza lo mismo para los datos de validacion y de prueba
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

#esta funcion realiza una vectorizacion del resultado
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e