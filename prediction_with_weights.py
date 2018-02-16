#!/usr/bin/python

import sys
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from helpers import *
from window_patch_extend import *
import numpy as np

batch_size = 32
img_size = 400
test_img_size = 608
window_size = 72
patch_size = 16
pred_dir = './predictions/'
mode = 'rgb'

def main():
    # print command line arguments
    if(len(sys.argv) < 2):
        print("not enought arguments, put the weight file as argument")
        return
    weight_path = sys.argv[1]

    print('from', weight_path)

    reg = 1e-6

    model = Sequential()

    model.add(Conv2D(32, (3, 3),  padding = 'same' ,input_shape=(window_size, window_size, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding = 'same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), padding = 'same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512 , kernel_regularizer=l2(reg)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(reg)))

    #opt = Adam(lr=0.001)
    #model.compile(loss='binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
    #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.2, patience=5, min_lr=0.0)

    model.load_weights(weight_path)

    nb_pred = 50
    pred_path = './test_set_images/test_'
    save_path = pred_dir

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    print('predicting')

    for i in range(1, nb_pred+1):
        print('  test_' + str(i))
        to_predict_img = load_image(pred_path+ str(i)+'/test_'+str(i)+'.png', mode)
        to_predict_windows = windows_from_img(to_predict_img, window_size, patch_size)
        to_predict_windows = np.asarray(to_predict_windows)
        pred = model.predict(to_predict_windows, batch_size)
        save_image(save_path + 'pred_' + str(i) + '.png', prediction_to_img(pred, test_img_size, patch_size, threshold=0.4))

if __name__ == "__main__":
    main()
