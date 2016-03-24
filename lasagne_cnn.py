# -*-  coding: utf-8 -*-
import lasagne
import theano
import theano.tensor as T
import gzip
import numpy as np
import sys
import time
author = 'liuqianchao'


def load_data_image(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)  # 与np.fromstring 常用来从文本中读取数字
    data = data.reshape(-1, 1, 28, 28)  # -1相当于缺省值
    return data / np.float64(256)


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


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def cnn(input_var=None):

    # input layer
    network = lasagne.layers.InputLayer(
            shape=(None, 1, 28, 28),
            input_var=input_var)
    # conv layer 1 32*5*5
    network = lasagne.layers.Conv2DLayer(
            network,
            num_filters=32,
            filter_size=(5, 5),
            stride=(1,1),
            pad=0,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform()
            )

    # pool layer 2
    network = lasagne.layers.MaxPool2DLayer(
            network,
            pool_size=(2,2)
            )

    # conv layer 3 32*5*5
    network = lasagne.layers.Conv2DLayer(
            network,
            num_filters=32,
            filter_size=(5, 5),
            stride=(1,1),
            pad=0,
            nonlinearity=lasagne.nonlinearities.rectify,
                                         )

    # fully-connected layer 4 of 256 units with 50% dropout
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

    # output layer, 10 units
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax
            )

    return network



if __name__ == "__main__":
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs[1] = sys.argv[1]

    num_epochs = 500
    print("Loading data...")
    # load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    input_var = T.tensor4('inputs')  # tensor4 means 4 dimension vector
    target_var = T.ivector('targets')  # a int data type vector

    print("Building model and compiling functions...")
    # build model
    network = cnn(input_var)

    # create loss function of training model
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction,target_var)
    loss = loss.mean()

    # update parameters
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=1e-2, momentum=0.9)

    # create loss function of test model
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
    test_lost = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1),target_var),dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates,on_unused_input='ignore')

    val_fn = theano.function([input_var, target_var], [test_loss,test_acc], on_unused_input='ignore')

    print "Strat training..."
    # training epoch
    for epoch in range(num_epochs):
        #each epoch train 500 pairs of data
        train_err = 0.0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs,targets = batch
            train_err += train_fn(inputs,targets)
            train_batches += 1

        # test validation data
        val_err = 0.0
        val_acc = 0.0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))