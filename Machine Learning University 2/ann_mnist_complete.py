# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:19:27 2019

@author: Natur
"""
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
from skimage import io
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers, metrics, regularizers, utils
from keras import backend as K
import matplotlib.pyplot as plt



batch_size = 512
num_classes = 10
epochs = 10
model_name='myMnistModel'
# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


img=x_train[0]
io.imshow(img)
plt.show()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(y_train[0])


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)



# convert class vectors to binary class matrices
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
print(y_train[0])

input_shape=(784,)

model = Sequential()
model.add(Dense(10,input_shape=input_shape))

model.add(Dense(28, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


#sgd = optimizers.SGD(lr=0.01, momentum=0.5)
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

model.summary()

history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))




# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


import sys
orig_stdout = sys.stdout
f = open(model_name+'_summary.txt', 'w')
sys.stdout = f
print(model.summary())
sys.stdout = orig_stdout
f.close()


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

