#!/usr/bin/env python
# coding: utf-8

# ### MNIST CONVOLUTIONAL NET
# #### Criação de rede convolucional com Keras embarcado no Tensorflow para classificação de números manuscritos.
# #### Treinamento com  dataset MNIST preexistente no pacote Keras.
# ##### Jeronimo Avelar Filho

# |--- Imported libraries (for model creation and training) ---|----------------------{{{
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.compat.v1 import ConfigProto
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL

# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
#---}}}


# |--- Examples for training and testing ---|----------------------{{{
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
#(train_images,train_labels), (test_images,test_labels) = mnist.load_data()
# No caso de vocês (imagem de ressonância: colocar as imagens dentro das estruturas train_images e test_images abaixo):
x = '/content/drive/MyDrive/TCC/dataset_completo/train'
x1 = '/content/drive/MyDrive/TCC/dataset_completo/test'
from tensorflow import keras

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    x,
    labels='inferred',
    label_mode='binary',
    seed=129,
    batch_size = 20,  #40
    image_size = (229,220),

)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    x1,
    labels='inferred',
    label_mode='binary',
    seed=129,
    batch_size = 20,
    image_size = (229,220),
)
Train_labels = []
Train_images = []

for images, labels in train_ds:
    for i in range(len(images)):
      Train_images.append(images[i])
      Train_labels.append(labels[i])
t_images = np.array(Train_images)
t_images = t_images.reshape(t_images.shape[0],229,220,3)
t_labels = np.array(Train_labels)

t_labels = t_labels.reshape(t_labels.shape[0],)

Test_labels = []
Test_images = []
for images, labels in val_ds:
    for i in range(len(images)):
      Test_images.append(images[i])
      Test_labels.append(labels[i])
te_images = np.array(Test_images)
te_images = te_images.reshape(te_images.shape[0],229,220,3)
te_labels = np.array(Test_labels)
te_labels = te_labels.reshape(te_labels.shape[0],)

(train_images,train_labels), (test_images,test_labels) = (t_images,t_labels),(te_images,te_labels)
##########################################################################
#train_images = train_images.reshape((te_images.shape[0], 229, 220, 1))  
train_images = train_images.astype('float32') / 255
arr_ = np.squeeze(train_images[1]) # you can give axis attribute if you wanna squeeze in specific dimension
plt.imshow(arr_)
plt.show()
arr_ = np.squeeze(train_images[2]) # you can give axis attribute if you wanna squeeze in specific dimension
plt.imshow(arr_)
plt.show()
#test_images = test_images.reshape((te_labels.shape[0], 229, 220, 1))
test_images = test_images.astype('float32') / 255
#arr_ = np.squeeze(test_images[1]) # you can give axis attribute if you wanna squeeze in specific dimension
#plt.imshow(arr_)
#plt.show()
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#for i in range (292):
#  print(train_labels[i])
print(train_labels[1])
print(test_labels[1])
##########################################################################################################################
#---}}}
from tensorflow.keras import optimizers
from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras

restnet = ResNet50(include_top=False, weights="imagenet", input_shape = (229, 220, 3))
for layer in restnet.layers:
    layer.trainable = False
restnet.summary()

print('Treinando o modelo...')
input_shape = (229, 220, 3)
model = models.Sequential()
model.add(restnet)

model.add(layers.Dense(200, activation='relu', input_dim=input_shape))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=100, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(2, activation='softmax'))
model.summary()

rms = optimizers.RMSprop(lr=0.001)

model.compile(loss='binary_crossentropy',optimizer=rms,metrics=[tf.keras.metrics.BinaryAccuracy(
    name="binary_accuracy")])
history = model.fit(train_images,train_labels, batch_size=100, epochs=300, verbose=1 , validation_split=0.3) #era30 batchsize
plt.plot(history.history['binary_accuracy'], label = 'Training', linewidth = 1.2)
plt.plot(history.history['val_binary_accuracy'], label = 'Validation', linewidth = 1.2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc="upper left")
plt.show()
plt.plot(history.history['loss'], label = 'Training', linewidth = 1.2)
plt.plot(history.history['val_loss'], label = 'Validation', linewidth = 1.2)
plt.xlabel('Epoch')
plt.ylabel('Loss function')
plt.legend(loc="upper left")
plt.show()
#---}}}

test_loss,test_acc = model.evaluate(test_images,test_labels)
print("Test Accuracy Validacao: ", test_acc)
print("Test Loss Validacao: ", test_loss)
