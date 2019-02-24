import pickle
import h5py
import numpy as np
import gzip

def load_muenster():
    path = "muenster_WGAN.pkl.gz"
    f = gzip.open(path, 'rb')
    [X_train, X_val, X_test, y_train, y_val, y_test] = pickle.load(f)
    f.close()
    X_train  = np.array(X_train)
    X_val  = np.array(X_val)
    X_test  = np.array(X_test)
    y_train  = np.array(y_train)
    y_val  = np.array(y_val)
    y_test  = np.array(y_test)
    # Data is already normalized

    # Convert to categorical
    y_train = bars_to_categorical(y_train)
    y_test = bars_to_categorical(y_test)
    y_val = bars_to_categorical(y_val)

    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test

def load_synthetic():
    # load data
    hf = h5py.File('X_data.h5', 'r')
    X = np.zeros((60000, 180, 180, 1), dtype='float32')
    hf['X'].read_direct(X)
    hf.close()
    hf = h5py.File('y_data.h5', 'r')
    y = np.zeros((60000, 13, 10, 1), dtype='float32')
    hf['y'].read_direct(y)
    hf.close()
    print(X.shape, y.shape)
    return X, y
