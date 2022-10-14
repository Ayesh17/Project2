import numpy as np


def perceptron_train(X,Y):
    print(X)
    print(Y)
    weights = np.zeros(len(X[0]))
    bias=0
    #do for 1D arrays too
    for i in range(len(X)):
        a=0

        for j in range(len(X[0])):
            a += X[i][j] * weights[j] + bias
        if (a * Y[i] <= 0):
            weights[j] = weights[j] + X[i][j]*Y[i]
            bias = bias + Y[i]
        print("iter", i, j)
        print("a", a)
        print("weights",weights)
        print("bias",bias)


    #a = wx+y

