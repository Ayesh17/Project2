import numpy as np

def perceptron_train(X,Y):
    print(X)
    print(Y)

    #do for 1D arrays too

    weights = np.zeros(len(X[0]))
    bias = 0
    count,weights, bias = epoch(X,Y,weights, bias)
    print(weights)
    print(bias)

    i=0
    while(count !=0):
        i+=1
        print("epoch", i)
        print()
        count,weights, bias = epoch(X,Y,weights, bias)
        print(weights)
        print(bias)


    #a = wx+y

def epoch(X,Y,weights,bias):

    count = 0

    for i in range(len(X)):
        a = 0
        for j in range(len(X[0])):
            a += X[i][j] * weights[j]
        a += bias
        print("a",a)
        if (a * Y[i] <= 0):
            # print("count", count)
            # print(Y[i])
            count += 1
            bias = bias + Y[i]
            for k in range(len(X[0])):
                weights[k] = weights[k] + X[i][k] * Y[i]

        # print("iter", i, j)
        # print("a", a)
        # print("weights", weights)
        # print("bias", bias)
        # print("count2", count)
    return count, weights, bias