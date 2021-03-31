import time

from bolsonaro.models.forest_pruning_sota import ForestPruningSOTA
from bolsonaro.models.utils import score_metric_mse, score_metric_indicator, aggregation_classification, aggregation_regression
from bolsonaro.utils import tqdm_joblib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from abc import abstractmethod, ABCMeta
import numpy as np
from scipy.stats import mode
from joblib import Parallel, delayed
from tqdm import tqdm


class KmeansForest(ForestPruningSOTA, metaclass=ABCMeta):
    """
    On extreme pruning of random forest ensembles for ral-time predictive applications', by Khaled Fawagreh, Mohamed Medhat Gaber and Eyad Elyan.
    """

    def _fit(self, X_train, y_train, X_val, y_val):
        self._base_estimator.fit(X_train, y_train)

        predictions_val = self._base_estimator_predictions(X_val).T
        predictions = self._base_estimator_predictions(X_train).T

        kmeans = KMeans(n_clusters=self._extracted_forest_size, random_state=self._models_parameters.seed).fit(predictions)
        labels = np.array(kmeans.labels_)

        # start_np_version = time.time()
        lst_pruned_forest = list()
        for cluster_idx in range(self._extracted_forest_size):  # pourrait être parallelise
            index_trees_cluster = np.where(labels == cluster_idx)[0]
            predictions_val_cluster = predictions_val[index_trees_cluster]  # get predictions of trees in cluster
            best_tree_index = self._get_best_tree_index(predictions_val_cluster, y_val)
            lst_pruned_forest.append(self._base_estimator.estimators_[index_trees_cluster[best_tree_index]])

        return lst_pruned_forest

    def _get_best_tree_index(self, y_preds, y_true):
        score = self._score_metric(y_preds, y_true)
        best_tree_index = self._best_score_idx(score)  # get best scoring tree (the one with lowest mse)
        return best_tree_index


class KMeansForestRegressor(KmeansForest, metaclass=ABCMeta):
    @staticmethod
    def init_estimator(model_parameters):
        return RandomForestRegressor(**model_parameters.hyperparameters,
                              random_state=model_parameters.seed, n_jobs=-1)

    def _aggregate(self, predictions):
        return aggregation_regression(predictions)

    def _score_metric(self, y_preds, y_true):
        return score_metric_mse(y_preds, y_true)

    @staticmethod
    def _best_score_idx(array):
        return np.argmin(array)

    @staticmethod
    def _worse_score_idx(array):
        return np.argmax(array)


class KMeansForestClassifier(KmeansForest, metaclass=ABCMeta):
    @staticmethod
    def init_estimator(model_parameters):
        return RandomForestClassifier(**model_parameters.hyperparameters,
                                                random_state=model_parameters.seed, n_jobs=-1)

    def _aggregate(self, predictions):
        return aggregation_classification(predictions)

    def _score_metric(self, y_preds, y_true):
        return score_metric_indicator(y_preds, y_true)

    def _selected_tree_predictions(self, X):
        predictions_0_1 = super()._selected_tree_predictions(X)
        predictions = (predictions_0_1 - 0.5) * 2
        return predictions

    def _base_estimator_predictions(self, X):
        predictions_0_1 = super()._base_estimator_predictions(X)
        predictions = (predictions_0_1 - 0.5) * 2
        return predictions

    @staticmethod
    def _best_score_idx(array):
        return np.argmax(array)

    @staticmethod
    def _worse_score_idx(array):
        return np.argmin(array)
