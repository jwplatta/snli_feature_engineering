import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, mutual_info_score, adjusted_rand_score
import time

class ValidateClusters:
    def __init__(self, model, dataset_name=None, dim_reduction=None, verbose=False, **kwargs):
        self.dataset_name = dataset_name
        self.dim_reduction = dim_reduction
        self.model = clone(model)
        self.results = {}


    def run(self, X, y_true):
        labels = self.model.fit_predict(X)
        self.labels = labels

        self.results['homogeneity'] = homogeneity_score(y_true, labels)
        self.results['completeness'] = completeness_score(y_true, labels)
        self.results['v_measure'] = v_measure_score(y_true, labels)
        self.results['mutual_info'] = mutual_info_score(y_true, labels)
        self.results['adjusted_rand'] = adjusted_rand_score(y_true, labels)

        if self.dataset_name:
            self.results['dataset'] = self.dataset_name

        if self.dim_reduction:
            self.results['dim reduction'] = self.dim_reduction

        return self.results
