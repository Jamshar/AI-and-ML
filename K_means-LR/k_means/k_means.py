import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, k):
        self.centroids = []
        self.labels = [] 
        self.clusters=[]
        self.k=k
        self.convert=0
    
    def get_centroids(self):
        centroids = [[]]*self.k
        for i in range(self.k):
            centroids[i]= np.average(self.clusters[i], axis = 0)
        
        for i in range(len(centroids)):
            centroids[i]=centroids[i].tolist()
        centroids=np.array(centroids)
        return centroids
    
    #Finding the total distance between new and old centroids.    
    def norm_centroids(self,old_centroid, new_centroid, k):
            dr = 0
            for i in range(k):
                dr = dr+np.linalg.norm(old_centroid[i]-new_centroid[i])
            return dr
        
    def euclidean_distance(self,x, y):
        return np.linalg.norm(x - y, ord=2, axis=-1)

    #Creating clusters based on the centroids.
    def make_clusters(self,X,k,Centroids):
        self.labels=[0]*len(X)
        clusters={}
        for i in range(k):
            clusters[i]=[]
        for i in range(len(X)):
            dr=[0]*k
            for j in range(k):
                dr[j] = euclidean_distance(X[i],Centroids[j])
            clusters[dr.index(min(dr))].append(X[i])
            self.labels[i] = dr.index(min(dr))
        return clusters
    
    #Implementing k-means++ algorithm.      
    def initial_centroids(self,X,k): 
        #Selecting first centroid at random.
        centroids = [X[np.random.randint(len(X))]]
        #Choosing the rest based on the distance from the centroids.
        
        for i in range(k-1):
            dist_list=[]
            for j in range(len(X)):
                D=[]
                #Instead of creating clusters based on each centroid, we
                #calculate the distance from all the points in the data
                #and store minimum to each centroid. Then chose a centroid 
                #that is proporsjonal to D(x). Here we chose the farthest point
                #from a centroid.
                for z in range(len(centroids)):
                    D.append([[euclidean_distance(X[j], centroids[z])],list(X[j])])
                dist_list.append(min(D))
            centroids.append(np.array(max(dist_list)[1])) 
        return centroids
    
    def fit(self, X):
        
        X = np.array(X).copy()
        #Step 1: choosing k random centroids
        self.clusters = []
        self.centroids=self.initial_centroids(X,self.k)
        
        #Step 2: Splitting the datapoints between the k centroids.
        #Step 3: Choosing new k centroids based on the center of the clusters.
        #Step 4:Repeating step 2 and 3 until the distance(epsilon) between k centrods is small.
        for i in range(self.k):
            epsilon=1
            while epsilon>0.0001:
                self.clusters= self.make_clusters(X,self.k,self.centroids)
                new_centroids = self.get_centroids()
                epsilon = self.norm_centroids(self.centroids, new_centroids, self.k)
                self.centroids=new_centroids  
         

    def predict(self, X):
        return np.array(self.labels)

def euclidean_distortion(X, z):
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()   
    return distortion

def euclidean_distance(x, y):
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])

def euclidean_silhouette(X, z):
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))