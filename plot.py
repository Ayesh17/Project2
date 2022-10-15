import clustering as kmeans
import perceptron as percept
import numpy as np
import matplotlib.pyplot as plt

#KMeans plot when K=2
def plot2(X,mu, K=2):
    # get cluster centers
    cluster_centers = kmeans.K_Means_better(X, K)

    # get euclidean distance
    distances = kmeans.distance(X, K, cluster_centers)

    # get clusters
    clusters = kmeans.get_clusters(X, K, distances)

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

    cluster_centers = kmeans.get_cluster_centers(clusters)

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


#KMeans plot when K=3
def plot3(X, mu, K=3):
    # get cluster centers
    cluster_centers = kmeans.K_Means_better(X, K)

    # get euclidean distance
    distances = kmeans.distance(X, K, cluster_centers)

    #get clusters
    clusters = kmeans.get_clusters(X, K, distances)

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

    cluster_centers = kmeans.get_cluster_centers(clusters)

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


#plot perceptron
def percept_plot(X,Y, w,b):
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

