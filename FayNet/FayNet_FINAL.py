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
from tensorflow.keras.models import Sequential, save_model
from tensorflow.compat.v1 import ConfigProto
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import optimizers


print('Iniciando o desenvolvimento da FayNet....')
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


# |--- Examples for training and testing ---|----------------------{{{
# from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
#(train_images,train_labels), (test_images,test_labels) = mnist.load_data()
# No caso de vocês (imagem de ressonância: colocar as imagens dentro das estruturas train_images e test_images abaixo):
data_dir = '/content/drive/MyDrive/TCC/dataset'
# x = 'C:/Users/Bezerra/Documents/UNB/TCC/Thiago/Machine learning/Testes_code/dataset_completo/train'
# x2 ='C:/Users/Bezerra/Documents/UNB/TCC/Thiago/Machine learning/Testes_code/dataset_completo/test'
image_size = (229,220)
batch_size = 8
##--------------PASTAS POR UM UNICO DIRETORIO-----------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="training",
    seed=500,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="validation",
    seed=200,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
)
###----PASTAS SEPARADAS EM TREINO E TESTE
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     x,
#     labels='inferred',
#     label_mode='binary',
#     seed=129,
#     image_size=image_size,
#     batch_size=batch_size,
#     color_mode="grayscale",

# )
# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     x2,
#     labels='inferred',
#     label_mode='binary',
#     seed=129,
#     image_size=image_size,
#     batch_size=batch_size,
#     color_mode="grayscale",
# )
###--------------------------------------------------
class_names = train_ds.class_names
print(class_names)

Train_labels = []
Train_images = []
for images, labels in train_ds:
    for i in range(len(images)):
      Train_images.append(images[i])
      Train_labels.append(labels[i])
t_images = np.array(Train_images)
t_images = t_images.reshape(t_images.shape[0],229,220,1) #229,220,1
t_labels = np.array(Train_labels)

t_labels = t_labels.reshape(t_labels.shape[0],)

Test_labels = []
Test_images = []
for images, labels in val_ds:   #val_ds ou test_ds
    for i in range(len(images)):
      Test_images.append(images[i])
      Test_labels.append(labels[i])
te_images = np.array(Test_images)
te_images = te_images.reshape(te_images.shape[0],229,220,1)
te_labels = np.array(Test_labels)
te_labels = te_labels.reshape(te_labels.shape[0],)



(train_images,train_labels), (test_images,test_labels) = (t_images,t_labels),(te_images,te_labels)
##########################################################################
train_images = train_images.reshape((1356, 229, 220, 1))  #1356# 469, 229, 220, 1((536, 229, 220, 1))
train_images = train_images.astype('float32') / 255
# arr_ = np.squeeze(train_images[1]) # you can give axis attribute if you wanna squeeze in specific dimension
# plt.imshow(arr_)
# plt.show()
test_images = test_images.reshape((580, 229, 220, 1)) # 580#201(134, 229, 220, 1) 100,100,1
test_images = test_images.astype('float32') / 255
# arr_ = np.squeeze(test_images[1]) # you can give axis attribute if you wanna squeeze in specific dimension
# plt.imshow(arr_)
# plt.show()
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

inputs = np.concatenate((train_images, test_images), axis=0)
targets = np.concatenate((train_labels, test_labels), axis=0)
##########################################################################################################################



model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5),strides=(1, 1), activation='relu', input_shape=(229, 220, 1)))
#model.add(layers.BatchNormalization())
#model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),strides=(1, 1), activation='relu'))
#model.add(layers.BatchNormalization())
#model.add(layers.AveragePooling2D((2,2)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=100, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(2, activation='softmax'))
#model.add(layers.Dense(2, activation='sigmoid'))
model.summary()


metrics = [tf.keras.metrics.TruePositives(thresholds=0.5, name='TP'), tf.keras.metrics.TrueNegatives(thresholds=0.5, name='TN'), tf.keras.metrics.FalsePositives(thresholds=0.5, name='FP'), tf.keras.metrics.FalseNegatives(thresholds=0.5, name='FN'), 'accuracy']
rms = optimizers.RMSprop(lr=0.0007)#tava 0,0007
sgd = optimizers.SGD(learning_rate=0.014, nesterov=True)
# |--- Model training ---|----------------------{{{
model.compile(loss='BinaryCrossentropy',optimizer=sgd,metrics =metrics)#['accuracy'] BinaryCrossentropy
history = model.fit(train_images, train_labels, batch_size=90 , epochs=150, verbose=1 , validation_split = 0.3 ) # bat 10 com 200 epocas

##-----print Metricas-------
results = model.evaluate(test_images,test_labels)
TP, TN, FP, FN, AC= results[1:]
print(TP, TN, FP, FN)

Acuracia = (TP+TN)/(TP+TN+FP+FN)

Precisao = TP/(TP+FP)
Especificidade = TN/(TN+FP)
Sensibilidade = TP/(TP+FN)

print("Acurácia: {:.5f}".format(Acuracia))
print("Precisão: {:.5f}".format(Precisao))
print("Especificidade: {:.5f}".format(Especificidade))
print("Sensibilidade: {:.5f}".format(Sensibilidade))

#-------------------------
plt.plot(history.history['accuracy'], label = 'Trainamento', linewidth = 1.2)
plt.plot(history.history['val_accuracy'], label = 'Validação', linewidth = 1.2)
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend(loc="upper left")
plt.show()
plt.plot(history.history['loss'], label = 'Trainamento', linewidth = 1.2)
plt.plot(history.history['val_loss'], label = 'Validação', linewidth = 1.2)
plt.xlabel('Épocas')
plt.ylabel('Função de perda')
plt.legend(loc="upper left")
plt.show()
#---}}}



# |--- Testing ---|----------------------{{{
# test_loss,test_acc = model.evaluate(test_images,test_labels)
# print("Test Accuracy Validacao: ", test_acc)
# print("Test Loss Validacao: ", test_loss)
# #---}}}

#------------salvando o modelo---
print('Salvando o modelo...')
model.save('fay.h5')
print('Modelo salvo!')

from sklearn.metrics import roc_curve, auc

y_pred = model.predict(test_images).ravel()
print(y_pred)   
print(len(y_pred))

y_test = test_labels.ravel()
print(len(y_test))
print(y_test)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('Taxa de falsos positivos')
plt.ylabel('Taxa de falsos negativos')
plt.title('Curva ROC')
plt.legend(loc='best')
plt.show()