from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from abc import abstractmethod, ABCMeta
import numpy as np
from tqdm import tqdm

from bolsonaro.models.utils import score_metric_mse, aggregation_regression, aggregation_classification, score_metric_indicator


class ForestPruningSOTA(BaseEstimator, metaclass=ABCMeta):

    def __init__(self, models_parameters):
        self._models_parameters = models_parameters
        self._extracted_forest_size = self._models_parameters.extracted_forest_size
        self._selected_trees = list()
        self._base_estimator = self.init_estimator(models_parameters)

    @staticmethod
    @abstractmethod
    def init_estimator(model_parameters):
        pass

    @abstractmethod
    def _fit(self, X_train, y_train, X_val, y_val):
        pass

    @property
    def models_parameters(self):
        return self._models_parameters

    @property
    def selected_trees(self):
        return self._selected_trees

    def fit(self, X_train, y_train, X_val, y_val):
        pruned_forest = self._fit(X_train, y_train, X_val, y_val)
        assert len(pruned_forest) == self._extracted_forest_size, "Pruned forest size isn't the size of expected forest: {} != {}".format(len(pruned_forest), self._extracted_forest_size)
        self._selected_trees = pruned_forest

    def _base_estimator_predictions(self, X):
        base_predictions = np.array([tree.predict(X) for tree in self._base_estimator.estimators_]).T
        return base_predictions

    def _selected_tree_predictions(self, X):
        base_predictions = np.array([tree.predict(X) for tree in self.selected_trees]).T
        return base_predictions

    def predict(self, X):
        predictions = self._selected_tree_predictions(X).T
        final_predictions = self._aggregate(predictions)
        return final_predictions

    def predict_base_estimator(self, X):
        return self._base_estimator.predict(X)

    def score(self, X, y):
        final_predictions = self.predict(X)
        score = self._score_metric(final_predictions, y)[0]
        return score

    @staticmethod
    @abstractmethod
    def _best_score_idx(array):
        """
        return index of best element in array

        :param array:
        :return:
        """
        pass


    @staticmethod
    @abstractmethod
    def _worse_score_idx(array):
        """
        return index of worse element in array

        :param array:
        :return:
        """
        pass


    @abstractmethod
    def _score_metric(self, y_preds, y_true):
        """
        get score of each predictors in y_preds

        y_preds.shape == (nb_trees, nb_sample)
        y_true.shape == (1, nb_sample)

        :param y_preds:
        :param y_true:
        :return:
        """
        pass

    @abstractmethod
    def _aggregate(self, predictions):
        """
        Aggregates votes of predictors in predictions

        predictions shape: (nb_trees, nb_samples)
        :param predictions:
        :return:
        """
        pass