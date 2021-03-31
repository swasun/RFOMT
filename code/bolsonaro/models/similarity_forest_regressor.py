import time

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from abc import abstractmethod, ABCMeta
import numpy as np
from tqdm import tqdm

from bolsonaro.models.forest_pruning_sota import ForestPruningSOTA
from bolsonaro.models.utils import score_metric_mse, aggregation_regression, aggregation_classification, score_metric_indicator


class SimilarityForest(ForestPruningSOTA, metaclass=ABCMeta):
    """
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2822360/
    """
    similarity_similarities = "similarity_similarities"
    similarity_predictions = "similarity_predictions"

    def _fit(self, X_train, y_train, X_val, y_val):
        self._base_estimator.fit(X_train, y_train)

        param = self._models_parameters.extraction_strategy

        # get score of base forest on val
        tree_list = list(self._base_estimator.estimators_)        # get score of base forest on val
        trees_to_remove = list()

        # get score of each single tree of forest on val
        val_predictions = self._base_estimator_predictions(X_val).T

        # boolean mask of trees to take into account for next evaluation of trees importance
        mask_trees_to_consider = np.ones(val_predictions.shape[0], dtype=bool)
        # the technique does backward selection, that is: trees are removed one after an other
        nb_tree_to_remove = len(tree_list) - self._extracted_forest_size
        with tqdm(range(nb_tree_to_remove), disable=True) as pruning_forest_bar:
            pruning_forest_bar.set_description(f'[Pruning forest s={self._extracted_forest_size}]')
            for _ in pruning_forest_bar:  # pour chaque arbre a extraire
                # get indexes of trees to take into account
                idx_trees_to_consider = np.arange(val_predictions.shape[0])[mask_trees_to_consider]
                val_predictions_to_consider = val_predictions[idx_trees_to_consider]
                nb_trees_to_consider = val_predictions_to_consider.shape[0]

                if param == self.similarity_predictions:
                    # this matrix has zero on the diag and 1/(L-1) everywhere else.
                    # When multiplying left the matrix of predictions (having L lines) by this zero_diag_matrix (square L), the result has on each
                    # line, the average of all other lines in the initial matrix of predictions
                    zero_diag_matrix = np.ones((nb_trees_to_consider, nb_trees_to_consider)) * (1 / (nb_trees_to_consider - 1))
                    np.fill_diagonal(zero_diag_matrix, 0)

                    leave_one_tree_out_predictions_val = zero_diag_matrix @ val_predictions_to_consider
                    leave_one_tree_out_predictions_val = self._activation(leave_one_tree_out_predictions_val)  # identity for regression; sign for classification
                    leave_one_tree_out_scores_val = self._score_metric(leave_one_tree_out_predictions_val, y_val)
                    # difference with base forest is actually useless
                    # delta_score = forest_score - leave_one_tree_out_scores_val

                    # get index of tree to remove
                    index_worse_tree = int(self._worse_score_idx(leave_one_tree_out_scores_val))

                elif param == self.similarity_similarities:
                    correlation_matrix = val_predictions_to_consider @ val_predictions_to_consider.T
                    average_correlation_by_tree = np.average(correlation_matrix, axis=1)

                    # get index of tree to remove
                    index_worse_tree = int(np.argmax(average_correlation_by_tree))  # correlation and MSE: both greater is worse

                else:
                    raise ValueError("Unknown similarity method {}. Should be {} or {}".format(param, self.similarity_similarities, self.similarity_predictions))

                index_worse_tree_in_base_forest = idx_trees_to_consider[index_worse_tree]
                trees_to_remove.append(tree_list[index_worse_tree_in_base_forest])
                mask_trees_to_consider[index_worse_tree_in_base_forest] = False
                pruning_forest_bar.update(1)

        pruned_forest = list(set(tree_list) - set(trees_to_remove))
        return pruned_forest

    @abstractmethod
    def _activation(self, leave_one_tree_out_predictions_val):
        pass



class SimilarityForestRegressor(SimilarityForest, metaclass=ABCMeta):

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

class SimilarityForestClassifier(SimilarityForest, metaclass=ABCMeta):

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
