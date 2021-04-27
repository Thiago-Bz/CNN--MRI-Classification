from tensorflow.keras.models import load_model
import cv2
import numpy as np
import glob

file_img = glob.glob('/content/drive/MyDrive/TCC/imagem_teste/**/*.png', recursive = True)

model = load_model('fay.h5')

model.compile(loss='binary_crossentropy',
              optimizer='sgd',#'sgd'
              metrics=['accuracy'])


n = 0
for file in file_img:
    n = n+1
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    # print(img.shape)
    img = cv2.resize(img,(229,220))
    img = np.reshape(img,[-1,229,220,1])
    classes = model.predict_classes(img)
    print (file)
    print ('Classe da imagem' + str(n), classes)
    



