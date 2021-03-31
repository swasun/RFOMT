from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from bolsonaro.models.model_parameters import ModelParameters
from bolsonaro.models.omp_forest import OmpForest, SingleOmpForest
from bolsonaro.models.omp_forest_classifier import OmpForestBinaryClassifier
from bolsonaro.utils import binarize_class_data, omp_premature_warning

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import OrthogonalMatchingPursuit
import warnings


class NonNegativeOmpForestBinaryClassifier(OmpForestBinaryClassifier):
    def predict(self, X, forest_size=None):
        """
        Make prediction.
        If forest_size is None return the list of predictions of all intermediate solutions

        :param X:
        :return:
        """
        forest_predictions = self._base_estimator_predictions(X)

        if self._models_parameters.normalize_D:
            forest_predictions /= self._forest_norms

        return self._omp.predict(forest_predictions, forest_size)

    def predict_no_weights(self, X, forest_size=None):
        """
        Make a prediction of the selected trees but without weight.
        If forest_size is None return the list of unweighted predictions of all intermediate solutions.

        :param X: some data to apply the forest to
        :return: a np.array of the predictions of the trees selected by OMP without applying the weight
        """
        forest_predictions = self._base_estimator_predictions(X)

        if forest_size is not None:
            weights = self._omp.get_coef(forest_size)
            select_trees = np.mean(forest_predictions[:, weights != 0], axis=1)
            return select_trees
        else:
            lst_predictions = []
            for sol in self._omp.get_coef():
                lst_predictions.append(np.mean(forest_predictions[sol != 0], axis=0))
            return lst_predictions


    def score(self, X, y, forest_size=None):
        """
        Evaluate OMPForestClassifer on (`X`, `y`).

        if Idx_prediction is None return the score of all sub forest.`

        :param X:
        :param y:
        :return:
        """
        # raise NotImplementedError("Function not verified")
        if forest_size is not None:
            predictions = self.predict(X, forest_size)
            # not sure predictions are -1/+1 so might be zero percent accuracy
            return np.sum(predictions != y) / len(y)
        else:
            predictions = self.predict(X)
            lst_scores = []
            for pred in predictions:
                lst_scores.append(np.sum(pred != y) / len(y))
            return lst_scores



if __name__ == "__main__":
    # X, y = load_boston(return_X_y=True)
    # X, y = fetch_california_housing(return_X_y=True)
    X, y = load_breast_cancer(return_X_y=True)
    y = (y-0.5)*2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.33, random_state = 42)

    # intermediate_solutions = [100, 200, 300, 400, 500, 1000]
    intermediate_solutions = [10, 20, 30, 40, 50, 100, 300]
    nnmodel_params = ModelParameters(extracted_forest_size=50,
                                     normalize_D=True,
                                     subsets_used=["train", "dev"],
                                     normalize_weights=False,
                                     seed=3,
                                     hyperparameters={
                                         "max_depth": 20,
                                         "min_samples_leaf": 1,
                                         "n_estimators": 1000,
                                         "max_features": "log2"
                                     },
                                     extraction_strategy="omp",
                                     non_negative=True,
                                     intermediate_solutions_sizes=intermediate_solutions)


    extracted_size = 300
    nn_ompforest = NonNegativeOmpForestBinaryClassifier(nnmodel_params)
    nn_ompforest.fit(X_train, y_train, X_train, y_train)
    model_params = ModelParameters(extracted_forest_size=extracted_size,
                    normalize_D=True,
                    subsets_used=["train", "dev"],
                    normalize_weights=False,
                    seed=3,
                    hyperparameters={
                        "max_depth": 20,
                        "min_samples_leaf": 1,
                        "n_estimators": 1000,
                        "max_features": "log2"
                    },
                    extraction_strategy="omp")
    omp_forest = OmpForestBinaryClassifier(model_params)
    omp_forest.fit(X_train, y_train, X_train, y_train)

    print("Breast Cancer")
    print("Score full forest on train", nn_ompforest.score_base_estimator(X_train, y_train))
    print("Score full forest on test", nn_ompforest.score_base_estimator(X_test, y_test))
    print("Size full forest", nnmodel_params.hyperparameters["n_estimators"])
    print("Size extracted forests", intermediate_solutions)
    print("Score non negative omp on train", nn_ompforest.score(X_train, y_train))
    print("Score non negative omp on test", nn_ompforest.score(X_test, y_test))
    print("Size extracted omp", extracted_size)
    print("Score omp on train", omp_forest.score(X_train, y_train))
    print("Score omp on test", omp_forest.score(X_test, y_test))
