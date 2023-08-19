import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples
import time

class GMMFindN:
    def __init__(self, n_range=np.arange(2, 10, 2), init_params='random_from_data', metric=False, verbose=False, **kwargs):
        self.n_range = n_range
        self.init_params = init_params
        self.fit_times = []
        self.bics = []
        self.aics = []
        self.labels = []
        self.log_likelihoods = []
        self.models = []

        self.metric = metric
        self.avg_silhouette_scores = []
        self.sample_silhouette_scores = []

        self.verbose = verbose


    def run(self, X):
        for n_comp in self.n_range:
            gmm = GaussianMixture(n_components=n_comp, covariance_type='full', init_params=self.init_params)
            start_time = time.time()
            gmm.fit(X)
            fit_time = time.time() - start_time
            self.fit_times.append(fit_time)

            start_time = time.time()
            labels = gmm.predict(X)
            self.labels.append(gmm.predict(X))
            self.bics.append(gmm.bic(X))
            self.aics.append(gmm.aic(X))
            self.log_likelihoods.append(gmm.score(X))
            self.models.append(gmm)
            score_time = time.time() - start_time

            if self.metric == 'silhouette':
                start_time = time.time()
                silh_score = silhouette_score(X, labels)
                self.avg_silhouette_scores.append(silh_score)
                sample_silh_scores = silhouette_samples(X, labels)
                self.sample_silhouette_scores.append(sample_silh_scores)
                end_time = time.time() - start_time

                if self.verbose:
                    print('\n*** Silhouette score calculation took {0} seconds.'.format(end_time))


            if self.verbose:
                print("\n------------\n{0}".format(gmm))
                print(
                    'n_components: {0} / BIC: {1} / AIC: {2} / log_likelihood: {3} / fit_time: {4}'.format(
                        n_comp, self.bics[-1], self.aics[-1], self.log_likelihoods[-1], fit_time
                    )
                )

