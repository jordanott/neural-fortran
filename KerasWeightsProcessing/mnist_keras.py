'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
import random
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import load_model
import pandas as pd

testcase=open('testcase.txt', 'r')
for line in testcase:
    line = line.strip().split()
    line = list(map(float, line))

df=pd.DataFrame(line).transpose()

model = load_model('keras10_io.h5')
print(model.summary())

pred = model.predict(df)
print(pred)













#
# # the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
# batch_size = 128
# num_classes = 10
# epochs = 20
# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
#
# model = Sequential()
# model.add(Dense(num_classes, activation='relu', input_shape=(784,)))
# # model.add(Dropout(0.2))
# # model.add(Dense(512, activation='relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(num_classes))
#
# model.summary()
#
# model.compile(loss='mse',
#               optimizer=RMSprop())
# # model.save("keras2.h5")
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(x_test, y_test))
# # score = model.evaluate(x_test, y_test, verbose=0)
# # print('Test loss:', score[0])
# # print('Test accuracy:', score[1])
#
# model.save('keras10_io.h5')
