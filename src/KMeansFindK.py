import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
import time

class KMeansFindK:
    def __init__(self, k_range=np.arange(2, 20, 2), init='k-means++', metric=None, verbose=False, **kwargs):
        self.k_range = k_range
        self.init = init
        self.labels = []

        self.models = []
        self.cluster_centers = []
        self.cluster_names = []
        self.fit_times = []
        self.wcss = []
        self.bcss = []
        self.cluster_sizes = []
        self.cluster_size_stds = []
        self.avg_intra_cluster_distances = []
        self.avg_inter_cluster_distances = []

        self.metric = metric
        self.avg_silhouette_scores = []
        self.sample_silhouette_scores = []
        self.davies_bouldin_scores = []
        self.calinski_harabasz_scores = []
        self.gap_values = []
        self.reference_distribution = np.array([])

        self.verbose = verbose


    def run(self, X):
        for n_clusters in self.k_range:
            kmeans = KMeans(n_clusters=n_clusters, init=self.init)

            start_time = time.time()
            kmeans.fit(X)
            fit_time = time.time() - start_time
            self.fit_times.append(fit_time)
            print('fit time {0}'.format(fit_time))

            wcss = kmeans.inertia_
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            self.cluster_centers.append(centroids)


            # NOTE: calculate the average intra-cluster sum squared errors for each cluster
            # avg_wcss = wcss / X.shape[0]
            self.labels.append(kmeans.labels_)
            self.cluster_names.append(np.unique(kmeans.labels_).tolist())
            self.wcss.append(wcss)

            # NOTE: compute BCSS
            bcss = []
            cluster_sizes = {}
            overall_mean = np.mean(X, axis=0) # NOTE: calculate the overall mean of the data
            for cluster in range(n_clusters):
                # STEP: get the number of points in the i-th cluster
                cluster_size = np.count_nonzero(kmeans.labels_ == cluster, axis=0)
                cluster_sizes[cluster] = cluster_size

                # STEP: get the i-th cluster centroid
                centroid = kmeans.cluster_centers_[cluster]

                # STEP: calculate the distance between the i-th cluster centroid and the overall mean
                dist = np.linalg.norm(centroid - overall_mean)

                # STEP: add the BCSS contribution from the i-th cluster to the total BCSS
                bcss.append(cluster_size * (dist ** 2))

            self.cluster_size_stds.append(np.std([c_size for _, c_size in cluster_sizes.items()]))
            self.cluster_sizes.append(cluster_sizes)
            self.bcss.append(np.sum(bcss))

            # NOTE: Compute the distances between each point and its centroid
            # distances = []
            # for i in range(X.shape[0]):
            #     centroid = centroids[labels[i]]
            #     distance = np.linalg.norm(X.iloc[i] - centroid)
            #     distances.append(distance)

            # distances = np.array(distances)

            # NOTE: Compute the intra-cluster distances
            # intra_cluster_distances = []
            # for label in labels:
            #     indices = np.where(labels == label)[0]
            #     distances_within_cluster = distances[indices]
            #     intra_cluster_distances.append(np.mean(distances_within_cluster))

            # self.avg_intra_cluster_distances.append(np.mean(intra_cluster_distances))

            # NOTE: Compute the inter-cluster distances
            inter_cluster_distances = []
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    distance = np.linalg.norm(centroids[i] - centroids[j])
                    inter_cluster_distances.append(distance)

            self.avg_inter_cluster_distances.append(np.mean(inter_cluster_distances))

            if self.metric == 'silhouette':
                start_time = time.time()
                silh_score = silhouette_score(X, kmeans.labels_)
                self.avg_silhouette_scores.append(silh_score)
                sample_silh_scores = silhouette_samples(X, kmeans.labels_)
                self.sample_silhouette_scores.append(sample_silh_scores)
                end_time = time.time() - start_time

                if self.verbose:
                    print('\n*** Silhouette score calculation took {0} seconds.'.format(end_time))
            elif self.metric == 'davies_bouldin':
                start_time = time.time()
                db_score = davies_bouldin_score(X, kmeans.labels_)
                end_time = time.time() - start_time
                self.davies_bouldin_scores.append(db_score)

                if self.verbose:
                    print('Davis-Bouldin score calculation took {0} seconds.'.format(end_time))
            elif self.metric == 'calinski_harabasz':
                start_time = time.time()
                ch_score = calinski_harabasz_score(X, kmeans.labels_)
                end_time = time.time() - start_time
                self.calinski_harabasz_scores.append(ch_score)

                if self.verbose:
                    print('Calinski-Harabasz score calculation took {0} seconds.'.format(end_time))
            elif self.metric == 'gap_statistic':
                ref_model = KMeans(n_clusters=n_clusters, init=self.init)
                ref_dist = self.__reference_distribution(X)
                ref_model.fit(ref_dist)
                self.gap_values.append(np.log(ref_model.inertia_) - np.log(wcss))

            self.models.append(kmeans)

            if self.verbose:
                print("\n------------\n{0}".format(kmeans))
                # print(
                #     'n_clusters: {0} / avg intra cluster distance: {1} / avg inter cluster dist: {2} / fit_time: {3}'.format(
                #         n_clusters, self.avg_intra_cluster_distances[-1], self.avg_inter_cluster_distances[-1], fit_time
                #     )
                # )
                print(
                    'n_clusters: {0} / avg inter cluster dist: {1} / fit_time: {2}'.format(
                        n_clusters, self.avg_inter_cluster_distances[-1], fit_time
                    )
                )
                if self.metric == 'silhouette':
                    print('Avg silhouette score: {0}'.format(self.avg_silhouette_scores[-1]))
                elif self.metric == 'davies_bouldin':
                    print('Davis-Bouldin index score: {0}'.format(self.davies_bouldin_scores[-1]))
                elif self.metric == 'calinski_harabasz':
                    print('Calinski-Harabasz index score: {0}'.format(self.calinski_harabasz_scores[-1]))
                elif self.metric == 'gap_statistic':
                    print('Gap value: {0}'.format(self.gap_values[-1]))

        return self.models


    def __reference_distribution(self, X):
        if self.reference_distribution.any():
            return self.reference_distribution
        else:
            self.reference_distribution = np.random.rand(*X.shape)
            return self.reference_distribution



