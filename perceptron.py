import numpy as np
import matplotlib.pyplot as plt

def perceptron_train(X,Y):

    weights = np.zeros(len(X[0]))
    bias = 0
    count,weights, bias = epoch(X,Y,weights, bias)

    #continue until a whole epoch goes without a change in weights and bias
    while(count !=0):
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

def plot(X,Y, w,b):
    # clf = Perceptron(n_iter=100).fit(X, Y)
    x1=w[0]
    x2=w[1]
    print(w[0])
    print(w[1])
    print(b)

    print(X)

    pos_X1_data = []
    pos_X2_data = []
    neg_X1_data = []
    neg_X2_data = []
    for i in range(len(X)):
        if (Y[i] == 1):
            pos_X1_data.append(X[i][0])
            pos_X2_data.append(X[i][1])
        else:
            neg_X1_data.append(X[i][0])
            neg_X2_data.append(X[i][1])


    plt.scatter(pos_X1_data, pos_X2_data, color="blue")
    plt.scatter(neg_X1_data, neg_X2_data, color="red")

    print("w",w)
    print("b",b)

    #w[0]*x1 + w[1]*x2 + b =0
    #w[1]*x2 = -w[0]*x1 -b
    #x2 = -(w[0]*x1 +b) /w[1]

    X1_points = np.zeros(7)
    X2_points = np.zeros(7)
    print(X1_points)
    print(X2_points)
    for i in [-3, -2, -1, 0, 1, 2, 3]:
        x1=i
        x2 = -(w[0] * x1 +b) / w[1]
        X1_points[i] = x1
        X2_points[i] = x2

    print(X1_points)
    print(X2_points)
    plt.plot(X1_points,X2_points, color="black")
    plt.show()

