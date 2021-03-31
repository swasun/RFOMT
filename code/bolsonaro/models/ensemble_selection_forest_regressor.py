import time

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from abc import abstractmethod, ABCMeta
import numpy as np
from tqdm import tqdm

from bolsonaro.models.forest_pruning_sota import ForestPruningSOTA
from bolsonaro.models.utils import score_metric_mse, aggregation_regression, aggregation_classification, score_metric_indicator


class EnsembleSelectionForest(ForestPruningSOTA, metaclass=ABCMeta):
    """
    'Ensemble selection from libraries of models' by Rich Caruana et al
    """

    def _fit(self, X_train, y_train, X_val, y_val):
        self._base_estimator.fit(X_train, y_train)

        val_predictions = self._base_estimator_predictions(X_val).T
        scores_predictions_val = self._score_metric(val_predictions, y_val)
        idx_best_score = self._best_score_idx(scores_predictions_val)

        lst_pruned_forest = [self._base_estimator.estimators_[idx_best_score]]

        nb_selected_trees = 1
        mean_so_far = val_predictions[idx_best_score]
        while nb_selected_trees < self._extracted_forest_size:
            # every new tree is selected with replacement as specified in the base paper

            # this matrix contains  at each line the predictions of the previous subset + the corresponding tree of the line
            # mean update formula: u_{t+1} = (n_t * u_t + x_t) / (n_t + 1)
            mean_prediction_subset_with_extra_tree = (nb_selected_trees * mean_so_far + val_predictions) / (nb_selected_trees + 1)
            predictions_subset_with_extra_tree = self._activation(mean_prediction_subset_with_extra_tree)
            scores_subset_with_extra_tree = self._score_metric(predictions_subset_with_extra_tree, y_val)
            idx_best_extra_tree = self._best_score_idx(scores_subset_with_extra_tree)
            lst_pruned_forest.append(self._base_estimator.estimators_[idx_best_extra_tree])

            # update new mean prediction
            mean_so_far = mean_prediction_subset_with_extra_tree[idx_best_extra_tree]
            nb_selected_trees += 1

        return lst_pruned_forest


    @abstractmethod
    def _activation(self, leave_one_tree_out_predictions_val):
        pass


class EnsembleSelectionForestClassifier(EnsembleSelectionForest, metaclass=ABCMeta):
    @staticmethod
    def init_estimator(model_parameters):
        return RandomForestClassifier(**model_parameters.hyperparameters,
                                    random_state=model_parameters.seed, n_jobs=-1)

    def _aggregate(self, predictions):
        return aggregation_classification(predictions)

    def _score_metric(self, y_preds, y_true):
        return score_metric_indicator(y_preds, y_true)

    def _activation(self, predictions):
        return np.sign(predictions)

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


class EnsembleSelectionForestRegressor(EnsembleSelectionForest, metaclass=ABCMeta):

    @staticmethod
    def init_estimator(model_parameters):
        return RandomForestRegressor(**model_parameters.hyperparameters,
                              random_state=model_parameters.seed, n_jobs=-1)

    def _aggregate(self, predictions):
        return aggregation_regression(predictions)

    def _score_metric(self, y_preds, y_true):
        return score_metric_mse(y_preds, y_true)

    def _activation(self, predictions):
        return predictions

    @staticmethod
    def _best_score_idx(array):
        return np.argmin(array)

    @staticmethod
    def _worse_score_idx(array):
        return np.argmax(array)
