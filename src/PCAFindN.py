import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import time

class PCAFindN:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.reconstruction_errors = []
        self.fit_times = []
        self.eigenvalues = np.array([])
        self.explained_variance_ratio = np.array([])


    def run(self, X):
        pca = PCA(n_components=self.n_components)
        pca.fit(X)

        for n in range(1, pca.n_components_+1):
            pca_n = PCA(n_components=n)

            X_transformed = pca_n.fit_transform(X)
            X_reconstructed = pca_n.inverse_transform(X_transformed)
            sq_error = (X - X_reconstructed)**2
            self.reconstruction_errors.append(sq_error.mean().mean())


        self.explained_variance_ratio = pca.explained_variance_ratio_
        self.eigenvalues = pca.explained_variance_

        return pca