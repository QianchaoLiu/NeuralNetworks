import cPickle as pickle
import numpy as np
import os
def load_CIFAR_batch(filename):
    with open(filename,'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']  # 10000 row of data, each row has 3*32*32 numbers
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
    return X,Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

if __name__ == "__main__":



    Xtr,Yri,Xte,Yte = load_CIFAR10('/Users/liuqianchao/Desktop/assignment2/cs231n/datasets/cifar-10-batches-py')
    print Xtr.shape

    print Xtr.transpose(0, 3, 1, 2)
    print Xtr.shape