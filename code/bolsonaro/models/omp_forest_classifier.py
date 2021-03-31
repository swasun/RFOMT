from bolsonaro.models.omp_forest import OmpForest, SingleOmpForest
from bolsonaro.utils import binarize_class_data, omp_premature_warning

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import OrthogonalMatchingPursuit
import warnings


class OmpForestBinaryClassifier(SingleOmpForest):

    DEFAULT_SCORE_METRIC = 'indicator'

    def __init__(self, models_parameters):
        estimator = RandomForestClassifier(**models_parameters.hyperparameters,
                                           random_state=models_parameters.seed, n_jobs=-1)
        super().__init__(models_parameters, estimator)

    def _check_classes(self, y):
        assert len(set(y).difference({-1, 1})) == 0, "Classes for binary classifier must be {-1, +1}"

    def fit(self, X_forest, y_forest, X_omp, y_omp, use_distillation=False):
        self._check_classes(y_forest)
        self._check_classes(y_omp)

        return super().fit(X_forest, y_forest, X_omp, y_omp, use_distillation=use_distillation)

    def _base_estimator_predictions(self, X):
        predictions_0_1 = super()._base_estimator_predictions(X)
        predictions = (predictions_0_1 - 0.5) * 2
        return predictions

    def score_base_estimator(self, X, y):
        predictions = self._base_estimator_predictions(X)
        evaluation = np.sum(np.sign(np.mean(predictions, axis=1)) == y) / len(y)
        return evaluation

    def predict_no_weights(self, X):
        """
        Apply the SingleOmpForest to X without using the weights.

        Make all the base tree predictions

        :param X: a Forest
        :return: a np.array of the predictions of the entire forest
        """

        forest_predictions = self._base_estimator_predictions(X)

        weights = self._omp.coef_
        omp_trees_predictions = forest_predictions[:, weights != 0]

        # Here forest_pred is the probability of being class 1.

        result_omp = np.mean(omp_trees_predictions, axis=1)

        return result_omp

    def score(self, X, y, metric=DEFAULT_SCORE_METRIC):
        """
        Evaluate OMPForestClassifer on (`X`, `y`) using `metric`

        :param X:
        :param y:
        :param metric: might be "indicator"
        :return:
        """
        predictions = self.predict(X)

        if metric == 'indicator':
            evaluation = np.abs(np.mean(np.abs(np.sign(predictions) - y) - 1))
        else:
            raise ValueError("Unsupported metric '{}'.".format(metric))

        return evaluation


class OmpForestMulticlassClassifier(OmpForest):

    DEFAULT_SCORE_METRIC = 'indicator'

    def __init__(self, models_parameters):
        estimator = RandomForestClassifier(**models_parameters.hyperparameters,
                                           random_state=models_parameters.seed, n_jobs=-1)
        super().__init__(models_parameters, estimator)
        # question: peut-être initialiser les omps dans le __init__? comme pour le SingleOmpForest
        self._dct_class_omp = {}

    def fit_omp(self, atoms, objective):
        assert len(self._dct_class_omp) == 0, "fit_omp can be called only once on {}".format(self.__class__.__name__)
        possible_classes = sorted(set(objective))
        for class_label in possible_classes:
            atoms_binary = binarize_class_data(atoms, class_label, inplace=False)
            objective_binary = binarize_class_data(objective, class_label, inplace=False)
            # TODO: peut etre considérer que la taille de forêt est globale et donc seulement une fraction est disponible pour chaque OMP...
            omp_class = OrthogonalMatchingPursuit(
                n_nonzero_coefs=self.models_parameters.extracted_forest_size,
                fit_intercept=True, normalize=False)

            with warnings.catch_warnings(record=True) as caught_warnings:
                # Cause all warnings to always be triggered.
                warnings.simplefilter("always")

                omp_class.fit(atoms_binary, objective_binary)

                # ignore any non-custom warnings that may be in the list
                caught_warnings = list(filter(lambda i: i.message != RuntimeWarning(omp_premature_warning), caught_warnings))

                if len(caught_warnings) > 0:
                    self._logger.error(f'number of linear dependences in the dictionary: {len(caught_warnings)}. model parameters: {str(self._models_parameters.__dict__)}')

            self._dct_class_omp[class_label] = omp_class
        return self._dct_class_omp

    def predict(self, X):
        '''forest_predictions = self._base_estimator_predictions(X)

        print(forest_predictions.shape)

        if self._models_parameters.normalize_D:
            forest_predictions /= self._forest_norms

        label_names = []
        preds = []
        for class_label, omp_class in self._dct_class_omp.items():
            label_names.append(class_label)
            atoms_binary = binarize_class_data(forest_predictions, class_label, inplace=False)
            print(atoms_binary.shape)
            preds.append(self._make_omp_weighted_prediction(atoms_binary, omp_class, self._models_parameters.normalize_weights))

        # TODO: verifier que ce n'est pas bugué ici

        preds = np.array(preds).T'''

        forest_predictions = np.array([tree.predict_proba(X) for tree in self._base_forest_estimator.estimators_]).T

        if self._models_parameters.normalize_D:
            forest_predictions /= self._forest_norms

        label_names = []
        preds = []
        num_class = 0
        for class_label, omp_class in self._dct_class_omp.items():
            label_names.append(class_label)
            atoms_binary = (forest_predictions[num_class] - 0.5) * 2 # centré réduit de 0/1 à -1/1
            preds.append(self._make_omp_weighted_prediction(atoms_binary, omp_class, self._models_parameters.normalize_weights))
            num_class += 1

        preds = np.array(preds).T
        max_preds = np.argmax(preds, axis=1)
        return np.array(label_names)[max_preds]

    def predict_no_weights(self, X):
        """
        Apply the SingleOmpForest to X without using the weights.

        Make all the base tree predictions

        :param X: a Forest
        :return: a np.array of the predictions of the entire forest
        """

        forest_predictions = np.array([tree.predict_proba(X) for tree in self._base_forest_estimator.estimators_]).T

        if self._models_parameters.normalize_D:
            forest_predictions = forest_predictions.T
            forest_predictions /= self._forest_norms
            forest_predictions = forest_predictions.T

        label_names = []
        preds = []
        num_class = 0
        for class_label, omp_class in self._dct_class_omp.items():
            weights = omp_class.coef_
            omp_trees_indices = np.nonzero(weights)
            label_names.append(class_label)
            atoms_binary = (forest_predictions[num_class].T - 0.5) * 2 # centré réduit de 0/1 à -1/1
            preds.append(np.sum(atoms_binary[omp_trees_indices], axis=0)/len(omp_trees_indices))
            num_class += 1

        preds = np.array(preds).T
        max_preds = np.argmax(preds, axis=1)
        return np.array(label_names)[max_preds]

    def score(self, X, y, metric=DEFAULT_SCORE_METRIC):
        predictions = self.predict(X)

        if metric == 'indicator':
            evaluation = np.sum(np.ones_like(predictions)[predictions == y]) / X.shape[0]
        else:
            raise ValueError("Unsupported metric '{}'.".format(metric))

        return evaluation

    @staticmethod
    def _make_omp_weighted_prediction(base_predictions, omp_obj, normalize_weights=False):
        if normalize_weights:
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


if __name__ == "__main__":
    forest = RandomForestClassifier(n_estimators=10)
    X = np.random.rand(10, 5)
    y = np.random.choice([-1, +1], 10)
    forest.fit(X, y)
    print(forest.predict(np.random.rand(10, 5)))
