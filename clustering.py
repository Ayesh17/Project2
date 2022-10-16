import numpy as np
import scipy.spatial as sci
import matplotlib.pyplot as plt

def min_max(X):
    min = np.min(X,axis=0)
    max = np.max(X, axis=0)
    return min, max

def K_Means(X, K, mu):
    # to suit both 1D and multi dimential arrays

    min,max = min_max(X)

    if (len(mu) == 0):
        if (isinstance(X[0], float)):
            mu = np.random.randint(min,max,K)
            mu = mu.reshape(len(mu), 1)
        else:
            mu = np.random.randint(min,max,size=(K,len(X[0])))


    if (isinstance(X[0], float)):
        X = X.reshape(len(X), 1)


    centers = K_Means_cal(X, K, mu)
    return centers


def K_Means_cal(X, K, mu):

    # get euclidean distance
    distances = distance(X, K, mu)

    # get clusters
    clusters = get_clusters(X, K, distances)

    empty = 0

    #Reduce num. of clusters if some clusters are empty
    indexes = []
    for i in range(len(clusters)):
        if (len(clusters[i]) == 0):
            indexes.append(i)
            empty +=1

    K -= len(indexes)

    for i in reversed( range(len(indexes))):
        clusters.pop(indexes[i])


    #get cluster centers
    centers = get_cluster_centers(clusters)

    # recursively call the function until cluster centers doesn't change
    if np.array_equal(mu, centers):
        print("centers", centers)
        return centers
    else:
        return K_Means_cal(X, K, centers)



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

def get_clusters(X, K, distances):
    arr = np.zeros(shape=(K, len(X)))

    for i in range(len(distances)):
        dist_list = distances[i]
        min = np.argmin(dist_list)
        arr[min][i] = i + 1

    #get clusters
    clusters = []
    for i in range(len(arr)):
        list = []
        for j in range(len(arr[i])):
            if (arr[i][j] != 0):
                pos = int(arr[i][j])
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

    #get 1000 random cluster centers
    centers = []
    for i in range (0,1000):
        arr = get_randoms(X,K)
        center = K_Means_cal(X,K,arr)
        centers.append(center)

    #get the count of cluster centers
    centers_list_arr = np.array(centers)

    val, count = np.unique(centers_list_arr, axis =0, return_counts = True)

    #get best cluster center
    best_center = val[count == count.max()]

    return best_center[0]




def get_randoms(X,K):
    #to remove duplicates to solve nan issues when same cluster center repeats
    dataset = X.tolist()
    result = [] #result is an array without duplicates
    for x in dataset:
        if x not in result:
            result.append(x)

    randoms = np.random.choice(len(result), size=K, replace=False)

    #to suit both 1D and multi dimential arrays
    if(isinstance(X[0],float)):
        arr = np.zeros(shape=(K, 1))
    else:
        arr = np.zeros(shape=(K, len(X[0])))

    # get random cluster centers
    for i in range(len(randoms)):
        arr[i] = result[randoms[i]]

    return arr
