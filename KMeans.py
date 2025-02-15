

import numpy as np

class KMeans:
    def __init__(self, n_clusters, random_state=0, retry=5, centroids=None):
        np.random.seed(random_state)
        self.n_clusters = n_clusters
        self.centroids = centroids
        self.retry = retry

    @staticmethod
    def euclidean_distance(datapoint, centroid):
        # Use np.linalg.norm for efficient distance calculation
        # ^^ this is better than using math.sqrt(sum((x-y)**2)) because it is vectorized :D
        return np.linalg.norm(datapoint.values[:-1] - centroid)

    @staticmethod
    def assign_centroids(x, centroids):
        # note to self: always use vectorized operations for efficiency DONT LOOP
        # calculate all distances between datapoints and centroids
        distances = np.array([[KMeans.euclidean_distance(row, centroid) for centroid in centroids] for _, row in x.iterrows()])
        # assign the closest centroid to each datapoint
        x['Cluster'] = np.argmin(distances, axis=1)
    
    @staticmethod
    def recalculate_centroids(x, centroids):
        for i in range(len(centroids)):
            relevent_points = x[x['Cluster'] == i]
            # it might be that not a single point is assigned to this centroid
            # in this case, we reassing a random point to this centroid 
            if relevent_points.shape[0] == 0:
                centroids[i][:] = x.drop('Cluster', axis=1).iloc[np.random.choice(x.shape[0])].values
            else:
                centroids[i][:] = relevent_points.drop('Cluster', axis=1).mean().values

    def train(self, x, max_iters=10000, verbose=False):
        
        x['Cluster'] = np.zeros(x.shape[0])

        for _ in range(self.retry):

            self.centroids = x.drop('Cluster', axis=1).iloc[np.random.choice(x.shape[0], self.n_clusters, replace=False)].values
            previous_centroids = self.centroids.copy()
            
            for _ in range(max_iters):
                KMeans.assign_centroids(x, self.centroids)
                KMeans.recalculate_centroids(x, self.centroids)
                if (self.centroids == previous_centroids).all():
                    break
                previous_centroids[:] = self.centroids

        if verbose:
            print('[INFO]: New "Cluser" column has been added to the data.')
            print('[INFO]: Training has been completed.')