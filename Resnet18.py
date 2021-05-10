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

print('Iniciando o desenvolvimento da Resnet18....')
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
    seed=500,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
)

class_names = train_ds.class_names
print(class_names)

Train_labels = []
Train_images = []
for images, labels in train_ds:
    for i in range(len(images)):
      Train_images.append(images[i])
      Train_labels.append(labels[i])
t_images = np.array(Train_images)
t_images = t_images.reshape(t_images.shape[0],229,220,1)
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
train_images = train_images.reshape((1356, 229, 220, 1))  #946# 469, 229, 220, 1((536, 229, 220, 1))
train_images = train_images.astype('float32') / 255
# arr_ = np.squeeze(train_images[1]) # you can give axis attribute if you wanna squeeze in specific dimension
# plt.imshow(arr_)
# plt.show()
test_images = test_images.reshape((580, 229, 220, 1)) # 402#201(134, 229, 220, 1)
test_images = test_images.astype('float32') / 255
# arr_ = np.squeeze(test_images[1]) # you can give axis attribute if you wanna squeeze in specific dimension
# plt.imshow(arr_)
# plt.show()
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

inputs = np.concatenate((train_images, test_images), axis=0)
targets = np.concatenate((train_labels, test_labels), axis=0)

## RESNET18 IMPLEMENTATION

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import tensorflow as tf


class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class ResNet18(Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out
    
model = ResNet18(2)
model.build(input_shape = (None,229,220,1)) 
rms = optimizers.RMSprop(lr=0.0001)
model.compile(optimizer = rms,loss='BinaryCrossentropy', metrics=["accuracy"]) 
model.summary()
history = model.fit(train_images, train_labels, batch_size=90 , epochs=100, verbose=1 , validation_split = 0.3 ) # bat 10 com 200 epocas

plt.plot(history.history['accuracy'], label = 'Training', linewidth = 1.2)
plt.plot(history.history['val_accuracy'], label = 'Validation', linewidth = 1.2)
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

# |--- Testing ---|----------------------{{{
test_loss,test_acc = model.evaluate(test_images,test_labels)
print("Test Accuracy Validacao: ", test_acc)
print("Test Loss Validacao: ", test_loss)
# #---}}}


