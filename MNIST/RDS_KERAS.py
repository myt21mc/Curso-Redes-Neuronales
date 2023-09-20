# -*- coding: utf-8 -*-
"""Untitled20.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aBOWuKVYpoFm1I31JJrX37ukDAI3UpLI

# Tarea 3. Red Densa Secuencial Keras
Redes Neuronales artificiales \\
Mytzi Yael Munguía Cuatlayotl \\
1.  Diseñar una red Densa secuencial (No convolucional) para clasificación de
dígitos e implementarla en Keras.

# Implementacion NN con Keras

## Carga y lectura de datos

Importamos las paqueterías
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras import regularizers
import numpy as np

"""Cargamos la base de datos mnist que ya está previamiente cargada"""

dataset=mnist.load_data()

"""Convertimos estos datos en arrays y dividimos en conjuntos de entrenamiento
y prueba.

"""

dat=np.array(dataset)
print(dat[1,1].shape)
(x_train, y_train), (x_test, y_test) = dataset

x_train[0]

dat

"""Imprimimos el dato [0,0]"""

import matplotlib.pyplot as plt
plt.imshow(dat[0,0][10000])

"""Tenemos 6000 datos de entrenamiento de 28x28"""

x_train.shape

x_trainv = x_train.reshape(60000, 784)
x_testv = x_test.reshape(10000, 784)
x_trainv = x_trainv.astype('float32') #Convertimos los números en decimales
x_testv = x_testv.astype('float32')

x_trainv /= 255  # x_trainv = x_trainv/255 normalización de cada entrada
x_testv /= 255

print(y_train[10000])

num_classes=10
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)

y_trainc[10000]

"""Aquí establecemos los mismos parámetros de la red de la tarea anterior."""

learning_rate = 3.0
epochs = 10
batch_size = 2

"""## a) Modelo semejante a la tarea anterior

**La primera red tendrá que ser equivalente a la que usaron en la tareaanterior.
Es decir, de la misma arquitectura, función de costo y optimizador.  En principio
se deberían obtener resultados semejantes, sin embargo la pregunta de este punto
es:  ¿Obtuviste resultados similares?, ¿Tardó lo mismo para entrenar el mismo número
de epocas?.En el reporte a entregar, hacer un comentario con respecto a las
cuestiones anteriores.  Hacer commit y subir a git-hub.  En el reporte y en TEAMS
subir el enlace al repositorio**

En el modelo de red neuronal anterior, definimos todas las funciones necesarias
para realizar el paso hacia delante, hacia atras y la actualizacion de parametros
con distintos optimizadores.

En el modelo basico usamos la siguiente configuracion:


*   Funcion de costo: MAE
*   Optimizador SGD
*   Funcion de activacion: Sigmoid
*   Learning rate: 3.0

**(Red previa a la implementación de cross-entropy con capa soft-max)**

"""

#Crear una red neuronal secuencial
model = Sequential()

#Agregar las capas a la red neuronal

#capa de entrada
model.add(Dense(784, activation='relu', input_shape=(784,)))

#capa intermedia
model.add(Dense(30, activation='sigmoid'))

#capa de salida
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

#Clasificacion
model.compile(loss='mse',optimizer= SGD(learning_rate=learning_rate), metrics=['accuracy'])

history = model.fit(x_trainv, y_trainc,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_testv, y_testc)
                    )

score = model.evaluate(x_testv, y_testc, verbose=1) #evaluar la eficiencia del modelo
print(score)
a=model.predict(x_testv) #predicción de la red entrenada
print(a.shape)
print(a[1])
print("resultado correcto:")
print(y_testc[1])

#Para guardar el modelo en disco
model0.save("red.h5")

#para cargar la red:
modelo_cargado = tf.keras.models.load_model('red.h5')

"""Vemos que existe una mejora de casi de 0.04 (En la red anterior se obtuvo 0.93
y en esta 0.97), a pesar de que son resultados similares la diferencia es muy
    significativa, el tiempo de entrenamiento fue mayor (lamentablemente no lo
pude comprobar, ya que las últimas modificaciones que le hice a la red de la
tarea anterior ya no me corrieron en la parte de la capa soft-max)

"""



"""## b) 3 Experimentos para intentar mejorar la eficiencia
Hacer 3 experimentos mas para intentar mejorar la eficiencia de lared.  Es decir,
aumenta capas o neuronas, puedes cambiar funcionesde activación y optimizador.  Es
cuestión de tu creatividad.  No usarregularización en este ejercicio.  En cada
experimento que hagas realiza un commit y sube el experimento a github con un
comentario explicando si mejoró la eficiencia de la red o no. En el reporte
explicarlos experimentos y comentar su eficiencia.

**Modelo 1**

Modificamos la activación para la capa intermedia, cambiamos sigmoid por relu,
al igual en la capa de salida cambiamos sigmoid por softmax. También se modificó
el tamaño de bache.\\
"""

model1 = Sequential()
model1.add(Dense(784, activation='relu', input_shape=(784,)))
model1.add(Dense(100, activation='relu'))
model1.add(Dense(num_classes, activation='softmax'))  # Clasificación multiclase, num_classes es el número de clases
model1.summary()

#Clasificacion
model1.compile(loss='mse',optimizer= SGD(learning_rate=learning_rate), metrics=['accuracy'])

history = model1.fit(x_trainv, y_trainc,
                    batch_size=10,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_testv, y_testc)
                    )

score = model1.evaluate(x_testv, y_testc, verbose=1) #evaluar la eficiencia del modelo
print(score)
a=model1.predict(x_testv) #predicción de la red entrenada
print(a.shape)
print(a[1])
print("resultado correcto:")
print(y_testc[1])

"""Podemos observar que no mejoro la eficiencia respecto al modelo anterior,
podemos ver que su eficiencia realmente es muy baja, la predicción es casi como
lanzar un dado. """