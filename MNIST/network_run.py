# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 14:37:13 2023

@author: mytzi
"""
import red_neuronal_mnist
import mnist

training_data, validation_data , test_data = mnist.load_data_wrapper() 
training_data = list(training_data)
test_data = list(test_data)

net = red_neuronal_mnist.Network([784,30,10])

net.SGD( training_data, 10, 2, 3.0, test_data=test_data)
