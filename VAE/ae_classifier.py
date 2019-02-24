import tensorflow as tf
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, BatchNormalization, Activation, Dropout, Conv2D, Conv2DTranspose
import keras.metrics as M
import keras.backend as K
import pickle
import h5py
import numpy as np
import gzip

batch_size=50
latent_dim=256

# gan_path = ""
#
# muenster_path = gan_path + "muenster_AE.pkl.gz"
#
# f = gzip.open(muenster_path, 'rb')
# [X_train, X_val, X_test, y_train, y_val, y_test] = pickle.load(f)
# f.close()
# X_train  = np.array(X_train)
# X_val  = np.array(X_val)
# X_test  = np.array(X_test)
# y_train  = np.array(y_train)
# y_val  = np.array(y_val)
# y_test  = np.array(y_test)
# # Data is already normalized
#
# print(X_train.shape, y_train.shape)
# print(X_val.shape, y_val.shape)
# print(X_test.shape, y_test.shape)
#
# y_train.astype(int)
# y_test.astype(int)
# y_val.astype(int)
# y_train = to_categorical(np.transpose(np.asarray(np.unravel_index(y_train, (10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10)))))
# y_test = to_categorical(np.transpose(np.asarray(np.unravel_index(y_test, (10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10)))))
# y_val = to_categorical(np.transpose(np.asarray(np.unravel_index(y_val, (10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10)))))

hf = h5py.File('X_train.h5', 'r')
X_train = np.zeros((54000, 180, 180, 1), dtype='float32')
hf['X_train'].read_direct(X_train)
hf.close()
hf = h5py.File('x_test.h5', 'r')
x_test = np.zeros((6000, 180, 180, 1), dtype='float32')
hf['x_test'].read_direct(x_test)
hf.close()
hf = h5py.File('B_train.h5', 'r')
y_train = np.zeros((54000, 1), dtype='int64')
hf['B_train'].read_direct(y_train)
hf.close()
hf = h5py.File('b_test.h5', 'r')
y_test = np.zeros((6000, 1), dtype='int64')
hf['b_test'].read_direct(y_test)
hf.close()

# CNN Option - One Hot
y_train = to_categorical(np.transpose(np.asarray(np.unravel_index(np.squeeze(y_train), (10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10)))))
y_test = to_categorical(np.transpose(np.asarray(np.unravel_index(np.squeeze(y_test), (10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10)))))

# # # FC option - numbers
# y_train = np.transpose(np.asarray(np.unravel_index(np.squeeze(y_train), (10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10))))
# y_test = np.transpose(np.asarray(np.unravel_index(np.squeeze(y_test), (10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10))))


encoder = keras.models.load_model('encoder')

X_latent = encoder.predict(X_train, batch_size=batch_size)
X_test_latent = encoder.predict(x_test, batch_size=batch_size)
# X_latent = X_latent.reshape(X_latent.shape[0], 16, 8, 16)


# def spars(y_true, y_pred):
#     # y_true (32,13) y_pred(32,13)
#     loss = 0
#     for i in range(batch_size):
#         one_hot_pred = K.one_hot(y_pred[i, :], 10)
#         loss += K.categorical_crossentropy(y_true[i, :, :], one_hot_pred)
#     return loss/batch_size
#
#
# def acc(y_true, y_pred):
#     acc = 0
#     for i in range(batch_size):
#         one_hot_pred = K.one_hot(y_pred[i, :], 10)
#         acc += M.binary_accuracy(y_true[i, :, :], one_hot_pred)
#     return acc/batch_size

# # CNN Option - one hot
# model = Sequential()
# model.add(Reshape((16, 16, 1), input_shape = (latent_dim,)))
# model.add(Conv2D(32, (3, 3), activation = 'relu'))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, (3, 3), activation = 'relu'))
# model.add(Dropout(0.25))
# model.add(Conv2D(128, (3, 3), activation = 'relu'))
# model.add(Flatten())
# model.add(Dense(512, activation = 'relu'))
# model.add(Dropout(0.25))
# model.add(Dense(130, activation = 'relu'))
# model.add(Reshape((13, 10)))
# model.add(Activation(activation = 'softmax'))
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc'])
# model.summary()

# FC Option
model = Sequential()
model.add(Dense(200, input_dim=latent_dim, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(170, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(130, activation= 'relu'))
model.add(Reshape((13,10)))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()



history = model.fit(X_latent, y_train, epochs = 200, batch_size = batch_size, validation_data = (X_test_latent, y_test), verbose = 2)