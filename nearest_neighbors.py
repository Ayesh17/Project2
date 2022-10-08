import numpy as np
import math
import scipy.spatial as sci

def KNN_test(X_train, Y_train, X_test, Y_test, K):

    #get Euclidean distances
    distances = []
    for i in range (len(X_test)):
        dist_list = []
        for j in range(len(X_train)):
            dist = sci.distance.euclidean(X_train[i],X_test[j])
            dist_list.append(dist)
        dist_list = np.argsort(dist_list)
        dist_list = dist_list[0:K]

        distances.append(dist_list)

    #get predictions
    preds = []
    for i in range(len(distances)):
        positives = 0
        negatives = 0
        for j in range(len(distances[i])):
            if (Y_test[distances[i][j]] == 1):
                positives += 1
            else:
                negatives +=1

        if( positives > negatives):
            preds.append(1)
        else:
            preds.append(-1)



    #get accuracy
    correct = 0
    wrong =0

    for i in range(len(Y_test)):
        if preds[i] == Y_test[i]:
            correct +=1
        else:
            wrong +=1

    accuracy = correct / (correct+wrong)
    return accuracy;

def choose_K(X_train, Y_train, X_val, Y_val):
    best_acc = 0;
    k = 0;

    #get the best K
    for i in range(1,len(X_train)):
        acc = KNN_test(X_train, Y_train, X_val, Y_val, i)
        if acc > best_acc:
            best_acc = acc
            k = i

    return k