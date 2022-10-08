import numpy as np
import math
import scipy.spatial as sci
from scipy import stats

def K_Means(X, K, mu):
    #get euclidean distance
    distances = distance(X, K, mu)

    #get clusters
    clusters = get_clusters(X, K, mu, distances)

    #get cluster centers
    centers = get_cluster_centers(clusters)
    #print(centers)

    #recursively call the function until cluster centers doesn't change
    if np.array_equal(mu,centers):
        #print(centers)
        return centers
    else:
        return K_Means(X, K, centers)




def distance(X, K, mu):
    cent = mu
    distances = []
    for i in range(len(X)):
        dist_list = []
        for j in range(K):
            dist = sci.distance.euclidean(X[i], cent[j])
            dist_list.append(dist)
        distances.append(dist_list)
    return distances

def get_clusters(X, K, mu, distances):
    arr = np.zeros(shape=(K, len(X)))

    for i in range(len(distances)):
        dist_list = distances[i]
        min = np.argmin(dist_list)
        arr[min][i] = i + 1

    clusters = []
    for i in range(len(arr)):
        list = []
        for j in range(len(arr[i])):
            if (arr[i][j] != 0):
                pos = int(arr[i][j])
                #print(X[pos-1])
                list.append(X[pos-1])
        clusters.append(list)

    return clusters

def get_cluster_centers(clusters):
    centers = []
    for i in range(len(clusters)):
        center= np.mean(clusters[i], axis=0)
        centers.append(center)
    return centers

def K_Means_better(X,K):

    centers = []
    for i in range (0,100):
        arr = get_randoms(X,K)
        center = K_Means(X,K,arr)
        centers.append(center)

    print(centers[0])
    centers_list = []
    for i in range(len(centers)):
        list = []
        for j in range(len(centers[i])):

            for k in range(len(centers[j])):
                list.append(centers[i][j][k])
        centers_list.append(list)
    print("t1")
    print(centers_list)
    print("t2")
    centers_list_arr = np.array(centers_list)
    print(centers_list_arr)
    val = np.unique(centers_list_arr, axis =0)
    print(val)
    #print(counts)

    mode = stats.mode(centers)
    print(mode)

def get_randoms(X,K):
    #to remove duplicates to solve nan issues when same cluster center repeats
    list= X.tolist()
    result = [] #result is an array without duplicates
    for x in list:
        if x not in result:
            result.append(x)


    randoms = np.random.choice(len(result), size=K, replace=False)
    arr = np.zeros(shape=(K, len(X[0])))
    for i in range(len(randoms)):
        arr[i] = result[randoms[i]]

    return arr