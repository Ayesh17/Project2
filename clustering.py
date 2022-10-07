import numpy as np
import math
import scipy.spatial as sci

def K_Means(X, K, mu):

    #cent = np.reshape(mu, (K,1))
    cent = mu
    distances = []
    for i in range(len(X)):
        dist_list = []
        for j in range(K):
            dist = sci.distance.euclidean(X[i], cent[j])
            dist_list.append(dist)
        # dist_list = np.argsort(dist_list)
        # dist_list = dist_list[0:K]

        distances.append(dist_list)
    clusters = [[None for _ in range(K)] for _ in range(len(X))]
    print(clusters)
    for i in range(len(distances)):
        dist_list = distances[i]
        dist_list = np.argsort(dist_list)
        clusters[i-1].append(dist_list)

        print (clusters)





    print(distances)
    return 0