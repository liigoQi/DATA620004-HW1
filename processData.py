from encodings import normalize_encoding
import numpy as np 
import struct

def load_data(image_path, label_path):
    with open(label_path, 'rb') as f:
        _, _ = struct.unpack('>II', f.read(8))
        label_data = np.fromfile(f, dtype=np.int8)

    with open(image_path, 'rb') as f:
        _, _, rows, cols = struct.unpack('>IIII', f.read(16))
        image_data = np.fromfile(f, dtype=np.uint8).reshape(len(label_data), rows, cols)
    
    return image_data, label_data 

def one_hot(labels):
    return np.eye(10)[labels]

def normalize(X):
    return 2 * (X / 255 - 0.5)

def data():
    path = './data/'
    image_train, label_train = load_data(path + 'train-images.idx3-ubyte', path + 'train-labels.idx1-ubyte')
    image_test, label_test = load_data(path + 't10k-images.idx3-ubyte', path + 't10k-labels.idx1-ubyte')

    input_dim = image_train.shape[-1] ** 2

    # normalize 
    image_train = normalize(image_train)
    image_test = normalize(image_test)
    
    # flatten
    image_train = image_train.reshape(len(image_train), input_dim)
    image_test = image_test.reshape(len(image_test), input_dim)
    
    # to one-hot 
    label_train = one_hot(label_train)
    label_test = one_hot(label_test)

    classes_num = 10

    # validation set: 10% of training set 
    valid_idx = int(0.9 * len(image_train))
    image_valid, label_valid = image_train[valid_idx:], label_train[valid_idx:]
    image_train, label_train = image_train[:valid_idx], label_train[:valid_idx]

    return image_train, label_train, image_valid, label_valid, image_test, label_test, classes_num, input_dim 

