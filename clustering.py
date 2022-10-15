import numpy as np
import scipy.spatial as sci
import matplotlib.pyplot as plt

def min_max(X):
    min = np.min(X,axis=0)
    max = np.max(X, axis=0)
    return min, max

def K_Means(X, K, mu):
    # to suit both 1D and multi dimential arrays
    print(len(mu))

    min,max = min_max(X)

    if (len(mu) == 0):
        if (isinstance(X[0], float)):
            mu = np.random.randint(min,max,K)
            mu = mu.reshape(len(mu), 1)
        else:
            mu = np.random.randint(min,max,size=(K,len(X[0])))

    print("mu new",mu[0][1])

    if (isinstance(X[0], float)):
        X = X.reshape(len(X), 1)


    centers = K_Means_cal(X, K, mu)
    return centers


def K_Means_cal(X, K, mu):

    # get euclidean distance
    distances = distance(X, K, mu)

    # get clusters
    clusters = get_clusters(X, K, distances)

    # get cluster centers
    centers = get_cluster_centers(clusters)

    # recursively call the function until cluster centers doesn't change
    if np.array_equal(mu, centers):
        return centers
    else:
        return K_Means_cal(X, K, centers)



def distance(X, K, mu):
    cent = mu
    distances = []
    print("X st",X)
    print("cent start", cent)
    for i in range(len(X)):
        dist_list = []
        for j in range(K):
            print("x",X[i])
            print("j",cent[j])
            dist = sci.distance.euclidean(X[i], cent[j])
            print("dist",dist)
            dist_list.append(dist)
        distances.append(dist_list)
    print("distances",distances)
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

    return best_center




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

def plot2(X,mu, K=2):
    # get euclidean distance
    distances = distance(X, K, mu)
    print(distances)

    # get clusters
    clusters = get_clusters(X, K, distances)

    cluster_centers = get_cluster_centers(clusters)
    print("cluster_centers",cluster_centers)

    cluster1= clusters[0]
    x_clust_1 = np.zeros(len(cluster1))
    y_clust_1 = np.zeros(len(cluster1))
    for i in range(len(clusters[0])):
        x_clust_1[i] = cluster1[i][0]
        y_clust_1[i]=cluster1[i][1]

    cluster2 = clusters[1]
    x_clust_2 = np.zeros(len(cluster2))
    y_clust_2 = np.zeros(len(cluster2))
    for i in range(len(cluster2)):
        x_clust_2[i] = cluster2[i][0]
        y_clust_2[i]=cluster2[i][1]

    plt.scatter(x_clust_1, y_clust_1, color ="blue")
    plt.scatter(x_clust_2, y_clust_2, color="black")

    cluster_centers = get_cluster_centers(clusters)
    print("cluster_centers", cluster_centers)
    x_clust_centers = np.zeros(len(cluster_centers))
    y_clust_centers = np.zeros(len(cluster_centers))
    for i in range(len(cluster_centers)):
        x_clust_centers[i] = cluster_centers[i][0]
        y_clust_centers[i] = cluster_centers[i][1]

    plt.scatter(x_clust_centers, y_clust_centers, color="red")

    n = ["C1", "C2"]
    for i, txt in enumerate(n):
        plt.annotate(txt, (x_clust_centers[i], y_clust_centers[i]))

    plt.show()

def plot3(X, mu, K=3):
    # get euclidean distance
    distances = distance(X, K, mu)
    print(distances)

    #get clusters
    clusters = get_clusters(X, K, distances)


    cluster1= clusters[0]
    x_clust_1 = np.zeros(len(cluster1))
    y_clust_1 = np.zeros(len(cluster1))
    for i in range(len(clusters[0])):
        x_clust_1[i] = cluster1[i][0]
        y_clust_1[i]=cluster1[i][1]

    cluster2 = clusters[1]
    x_clust_2 = np.zeros(len(cluster2))
    y_clust_2 = np.zeros(len(cluster2))
    for i in range(len(cluster2)):
        x_clust_2[i] = cluster2[i][0]
        y_clust_2[i]=cluster2[i][1]\

    cluster3 = clusters[2]
    x_clust_3 = np.zeros(len(cluster3))
    y_clust_3 = np.zeros(len(cluster3))
    for i in range(len(cluster3)):
        x_clust_3[i] = cluster3[i][0]
        y_clust_3[i] = cluster3[i][1]

    plt.scatter(x_clust_1, y_clust_1, color ="green")
    plt.scatter(x_clust_2, y_clust_2, color="black")
    plt.scatter(x_clust_3, y_clust_3, color="blue")


    cluster_centers = get_cluster_centers(clusters)
    print("cluster_centers", cluster_centers)
    x_clust_centers = np.zeros(len(cluster_centers))
    y_clust_centers = np.zeros(len(cluster_centers))
    for i in range(len(cluster_centers)):
        x_clust_centers[i] = cluster_centers[i][0]
        y_clust_centers[i] = cluster_centers[i][1]

    plt.scatter(x_clust_centers, y_clust_centers, color="red")

    n = ["C1","C2","C3"]
    for i, txt in enumerate(n):
        plt.annotate(txt, (x_clust_centers[i], y_clust_centers[i]))


    plt.show()