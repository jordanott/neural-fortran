'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''
import keras
import random
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import load_model
import numpy as np


testcase=open('testcase.txt', 'r')
for line in testcase:
    line = line.strip().split()
    line = list(map(float, line))

x = np.array(line)[np.newaxis]
print 'Data:', x.shape
num_classes = 10

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(num_classes))

model.summary()

model.compile(loss='mse',optimizer=RMSprop())
# model = load_model('keras10_io.h5')

print model.predict(x)
model.save('keras10_io.h5')
