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

    x_arr = np.zeros(7)
    y_arr = np.zeros(7)
    print(x_arr)
    print(y_arr)
    for i in [-3,-2,-1,0,1,2,3]:
        x = i
        if(i == 0):
            y = 0
        else:
            y = (-x2*i - b)/i
        x_arr[i]=x
        y_arr[i]=y

    print(x_arr)
    print(y_arr)

    X = x_arr
    Y= y_arr

    Z = X*x1 + Y *x2 + b
    plt.contour(X, Y, Z, colors='black');
    plt.show()


    # plt.plot(x_arr,y_arr)
    # plt.show()

    # h=.02
    # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                      np.arange(y_min, y_max, h))
    #
    # # Plot the decision boundary. For that, we will assign a color to each
    # # point in the mesh [x_min, m_max]x[y_min, y_max].
    # fig, ax = plt.subplots()
    # Z = predict(X, w,b)
    #
    # # Put the result into a color plot
    # # Z = Z.reshape(xx.shape)
    # ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    # ax.axis('off')
    #
    # # Plot also the training points
    # ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    #
    # ax.set_title('Perceptron')