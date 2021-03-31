from bolsonaro import LOG_PATH
from bolsonaro.error_handling.logger_factory import LoggerFactory
from bolsonaro.models.nn_omp import NonNegativeOrthogonalMatchingPursuit
from bolsonaro.utils import omp_premature_warning

from abc import abstractmethod, ABCMeta
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.base import BaseEstimator
import warnings


class OmpForest(BaseEstimator, metaclass=ABCMeta):

    def __init__(self, models_parameters, base_forest_estimator):
        self._base_forest_estimator = base_forest_estimator
        self._models_parameters = models_parameters
        self._logger = LoggerFactory.create(LOG_PATH, __name__)
        self._selected_trees = list()

    @property
    def models_parameters(self):
        return self._models_parameters

    def predict_base_estimator(self, X):
        return self._base_forest_estimator.predict(X)


    def _base_estimator_predictions(self, X):
        base_predictions = np.array([tree.predict(X) for tree in self._base_forest_estimator.estimators_]).T
        return base_predictions

    @property
    def forest(self):
        return self._base_forest_estimator.estimators_

    # sklearn baseestimator api methods
    def fit(self, X_forest, y_forest, X_omp, y_omp, use_distillation=False):
        # print(y_forest.shape)
        # print(set([type(y) for y in y_forest]))
        self._base_forest_estimator.fit(X_forest, y_forest)
        self._extract_subforest(X_omp,
            self.predict_base_estimator(X_omp) if use_distillation else y_omp) # type: OrthogonalMatchingPursuit
        return self

    def _extract_subforest(self, X, y):
        """
        Given an already estimated regressor: apply OMP to get the weight of each tree.

        The X data is used for interrogation of every tree in the forest. The y data
        is used for finding the weights in OMP.

        :param X: (n_sample, n_features) array
        :param y: (n_sample,) array
        :return:
        """
        self._logger.debug("Forest make prediction on X")
        D = self._base_estimator_predictions(X)

        if self._models_parameters.normalize_D:
            # question: maybe consider other kinds of normalization.. centering?
            self._logger.debug("Compute norm of predicted vectors on X")
            self._forest_norms = np.linalg.norm(D, axis=0)
            D /= self._forest_norms

        self._logger.debug("Apply orthogonal maching pursuit on forest for {} extracted trees."
                           .format(self._models_parameters.extracted_forest_size))

        self.fit_omp(D, y)

    @staticmethod
    def _make_omp_weighted_prediction(base_predictions, omp_obj, normalize_weights=False):
        if normalize_weights:
            raise ValueError("Normalize weights is deprecated")
            # we can normalize weights (by their sum) so that they sum to 1
            # and they can be interpreted as impact percentages for interpretability.
            # this necessits to remove the (-) in weights, e.g. move it to the predictions (use unsigned_coef) --> I don't see why

            # question: je comprend pas le truc avec nonszero?
            # predictions = self._omp.predict(forest_predictions) * (1 / (np.sum(self._omp.coef_) / len(np.nonzero(self._omp.coef_))))
            coef_signs = np.sign(omp_obj.coef_)[np.newaxis, :]  # add axis to make sure it will be broadcasted line-wise (there might be a confusion when forest_prediction is square)
            unsigned_coef = (coef_signs * omp_obj.coef_).squeeze()
            intercept = omp_obj.intercept_

            adjusted_forest_predictions = base_predictions * coef_signs
            predictions = adjusted_forest_predictions.dot(unsigned_coef) + intercept

        else:
            predictions = omp_obj.predict(base_predictions)

        return predictions

    @abstractmethod
    def fit_omp(self, atoms, objective):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def score(self, X, y):
        pass

class SingleOmpForest(OmpForest):

    def __init__(self, models_parameters, base_forest_estimator):
        if models_parameters.extraction_strategy == 'omp_nn':
            self._omp = NonNegativeOrthogonalMatchingPursuit(
                max_iter=models_parameters.extracted_forest_size,
                intermediate_solutions_sizes=models_parameters.intermediate_solutions_sizes,
                fill_with_final_solution=True
            )
        else:
            # fit_intercept shouldn't be set to False as the data isn't necessarily centered here
            # normalization is handled outsite OMP
            self._omp = OrthogonalMatchingPursuit(
                n_nonzero_coefs=models_parameters.extracted_forest_size,
                fit_intercept=True, normalize=False)

        super().__init__(models_parameters, base_forest_estimator)

    def fit_omp(self, atoms, objective):
        self._omp.fit(atoms, objective)

        """with warnings.catch_warnings(record=True) as caught_warnings:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            

            # ignore any non-custom warnings that may be in the list
            caught_warnings = list(filter(lambda i: i.message != RuntimeWarning(omp_premature_warning), caught_warnings))

            if len(caught_warnings) > 0:
                self._logger.error(f'number of linear dependences in the dictionary: {len(caught_warnings)}. model parameters: {str(self._models_parameters.__dict__)}')"""

    def predict(self, X):
        """
        Apply the SingleOmpForest to X.

        Make all the base tree predictions then apply the OMP weights for pruning.

        :param X:
        :return:
        """
        forest_predictions = self._base_estimator_predictions(X)

        if self._models_parameters.normalize_D:
            forest_predictions /= self._forest_norms

        return self._make_omp_weighted_prediction(forest_predictions, self._omp, self._models_parameters.normalize_weights)

    def predict_no_weights(self, X):
        """
        Apply the SingleOmpForest to X without using the weights.

        Make all the base tree predictions

        :param X: a Forest
        :return: a np.array of the predictions of the trees selected by OMP without applying the weight
        """
        forest_predictions = np.array([tree.predict(X) for tree in self._base_forest_estimator.estimators_])

        weights = self._omp.coef_
        select_trees = np.mean(forest_predictions[weights != 0], axis=0)
        return select_trees
