import numpy as np

def perceptron_train(X,Y):
    print(X)
    print(Y)

    #do for 1D arrays too

    weights = np.zeros(len(X[0]))
    bias = 0
    count,weights, bias = epoch(X,Y,weights, bias)
    while(count !=0):
        count,weights, bias = epoch(X,Y,weights, bias)

    return weights,bias
    #a = wx+y

def epoch(X,Y,weights,bias):
    count = 0
    for i in range(len(X)):
        a = 0
        for j in range(len(X[0])):
            a += X[i][j] * weights[j]
        a += bias
        if (a * Y[i] <= 0):
            count += 1
            bias = bias + Y[i]
            for k in range(len(X[0])):
                weights[k] = weights[k] + X[i][k] * Y[i]
    return count, weights, bias

def perceptron_test(X_test,Y_test,w,b):

    return 0