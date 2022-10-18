<<<<<<< Updated upstream
=======
import numpy as np
import matplotlib.pyplot as plt

def perceptron_train(X,Y):

    weights = np.zeros(len(X[0]))
    bias = 0
    count,weights, bias = epoch(X,Y,weights, bias)
    epoch_count = 1

    #continue until a whole epoch goes without a change in weights and bias or epoch count go higher than 1000
    while(count !=0) and (epoch_count <=1000):
        epoch_count +=1
        count,weights, bias = epoch(X,Y,weights, bias)

    return weights,bias

#get weights and bias at the end of an epoch
def epoch(X,Y,weights,bias):
    #count of how many times weights changed
    count = 0
    for i in range(len(X)):
        a = 0
        for j in range(len(X[0])):
            a += X[i][j] * weights[j]
        a += bias

        if (a * Y[i] <= 0):
            count += 1
            bias = bias + Y[i]
            #update weights
            for k in range(len(X[0])):
                weights[k] = weights[k] + X[i][k] * Y[i]
    return count, weights, bias


def perceptron_test(X_test,Y_test,w,b):
    preds= predict(X_test,w,b)

    #Calculate accuracy
    correct =0
    wrong =0
    for i in range(len(Y_test)):
        if preds[i] == Y_test[i]:
            correct += 1
        else:
            wrong +=1
    accuracy = correct / (correct + wrong)

    return accuracy

def predict(X_test,w,b):
    preds = []

    # get predictions
    for i in range(len(X_test)):
        a = 0
        for j in range(len(X_test[i])):
            a += X_test[i][j] * w[j]
        a += b

        if (a >= 1):
            a = 1
        else:
            a = -1
        preds.append(a)
    return preds


>>>>>>> Stashed changes
