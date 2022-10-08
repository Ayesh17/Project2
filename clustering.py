import numpy as np
import math
import scipy.spatial as sci

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
        print (centers)
        return centers
    else:
        K_Means(X, K, centers)







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