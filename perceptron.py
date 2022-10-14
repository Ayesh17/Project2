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
    preds=[]
    for i in range(len(X_test)):
        a=0
        for j in range(len(X_test[i])):
            a += X_test[i][j]*w[j]
        a+=b

        if(a>=1):
            a = 1
        else:
            a = -1
        preds.append(a)

    ##Calculate accuracy
    correct =0
    wrong =0
    for i in range(len(Y_test)):
        if preds[i] == Y_test[i]:
            correct += 1
        else:
            wrong +=1
    accuracy = correct / (correct + wrong)

    return accuracy