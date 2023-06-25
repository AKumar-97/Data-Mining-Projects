import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

class KMeansImp():
    def __init__(self, X_Data, K_Clusters, F_random):
        self.K_Clusters = K_Clusters
        self.X_Data = X_Data
        self.Y = None
        self.errorValue = None
        self.distValue = np.zeros((X_Data.shape[0], K_Clusters))
        self.Centroid = np.zeros((K_Clusters, X_Data.shape[1]))
        self.featurez = F_random

    def randomCentroid(self):
        centroid = np.random.choice(self.X_Data.shape[0],1)
        self.Centroid[0] = self.X_Data[centroid]
        for i in range(1, self.K_Clusters):
            distanceMeasure = euclidean_distances(self.X_Data, self.Centroid[0:i])
            #distanceMeasure = cosine_similarity(self.X_Data, self.Centroid[0:i])
            self.Centroid[i] = self.X_Data[np.argmax(np.amin(distanceMeasure, axis = 1))]

    def CentroidComp(self):
        CentroidList = np.zeros((self.K_Clusters, self.featurez))
        for i in range(self.K_Clusters):
            CentroidList[i] = self.X_Data[self.Y == i].mean(axis = 0)

        self.Centroid = np.asarray(CentroidList)

    def DistanceComp(self):
        self.distValue = euclidean_distances(self.X_Data,self.Centroid)

    def ClusterComp(self):
        self.Y = np.argmin(self.distValue, axis = 1)

    def errorComp(self):
        errList = np.zeros((self.K_Clusters))
        for i in range(self.K_Clusters):
            errList[i] = (np.min(self.distValue[self.Y == i], axis =1)).sum()

        self.errorValue = np.asarray(errList)

    def Main(self, itr):
        for i in range(itr):
            self.DistanceComp()
            self.ClusterComp()
            self.CentroidComp()
            self.errorComp()
        
        return self.Centroid, self.errorValue, self.Y