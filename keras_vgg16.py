from keras import models

from keras import optimizers
from keras.applications import VGG16
import keras.backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Sequential
import matplotlib.pyplot as plt
import cv2 as cv
from keras.layers.core import Flatten, Dense, Dropout

def iou_coef(y_true, y_pred, smooth=1):
    """
    IoU = (|X &amp; Y|)/ (|X or Y|)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum((y_true,-1) + K.sum(y_pred,-1) - intersection)
    return (intersection + smooth) / ( union + smooth)
def iou(true, pred):
    intersection = true * pred

    notTrue = 1 - true
    union = true + (notTrue * pred)

    return K.sum(intersection) / K.sum(union)
if __name__ == "__main__":
    n_train=27088
    filename=[]
    labels=np.zeros(shape=(n_train,4))
    sizes=np.zeros(shape=(n_train,2))
    with open("./dataset2.txt", "r") as ins:
        infoFile = ins.readlines()  # reading lines from file
        c = 0
        for line in infoFile:  # reading line by line

            words = line.split(' ')
            filename.append(words[0])

            labels[c]=[words[1],words[2],words[3],words[4]]
            sizes[c]=[words[5],words[6]]
            c=c+1
    print('labels done')
    model = VGG16(weights='imagenet', include_top=False)
    print(labels[1][0]+1)
    model.summary()
    train_feature=np.zeros(shape=(n_train,7,7,512))

    c=0
    for i in filename:
        img_path =i

        img = cv.imread(img_path)
        img_data=cv.resize(img,(224,224))

        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        vgg16_feature = model.predict(img_data)

        train_feature[c]=vgg16_feature


    print('extracting features done')

    model = Sequential()
    model.add(Flatten(input_shape=train_feature.shape[1:]))
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(4, activation='linear'))
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=iou,optimizer=sgd, metrics=[iou])


    history=model.fit(train_feature, labels, epochs=3,batch_size=32)
    prediction=model.predict(train_feature,labels)
    plt.plot(history.history[iou])
    plt.title('loss graph')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    for i in range(10):
        print('prediction:',prediction[i])
        print('ground truth:',labels[i])
    score = model.evaluate(train_feature, labels, batch_size=128)

    plt.show()
