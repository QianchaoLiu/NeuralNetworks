# -*-  coding: utf-8 -*-
import lasagne
import theano
import theano.tensor as T
import gzip
import numpy as np
import sys
author = 'liuqianchao'


def load_data_image(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)  # 与np.fromstring 常用来从文本中读取数字
    data.reshape(-1, 1, 28, 28)  # -1相当于缺省值
    return data/ np.float64(256)

def load_data_label(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def load_data():
    X_train = load_data_image('/Users/liuqianchao/Desktop/Lasagne/examples/train-images-idx3-ubyte.gz')
    y_train = load_data_label('/Users/liuqianchao/Desktop/Lasagne/examples/train-labels-idx1-ubyte.gz')
    X_test = load_data_image('/Users/liuqianchao/Desktop/Lasagne/examples/t10k-images-idx3-ubyte.gz')
    y_test = load_data_label('/Users/liuqianchao/Desktop/Lasagne/examples/t10k-labels-idx1-ubyte.gz')

    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    return X_train, y_train, X_val, y_val, X_test, y_test

def cnn(input_val=None):

    network = lasagne.layers.InputLayer()


if __name__ == "__main__":
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs[1] = sys.argv[1]