import numpy as np

from bolsonaro.models.model_parameters import ModelParameters
from bolsonaro.models.omp_forest_classifier import OmpForestBinaryClassifier, OmpForestMulticlassClassifier
from bolsonaro.models.omp_forest_regressor import OmpForestRegressor


def test_binary_classif_omp():

    model_parameters = ModelParameters(
        1, False, ['train+dev', 'train+dev'], False, 1,
        {'n_estimators': 100}, 'omp'
    )

    omp_forest = OmpForestBinaryClassifier(model_parameters)
    X_train = [[1, 0], [0, 1]]
    y_train = [-1, 1]

    omp_forest.fit(X_train, y_train, X_train, y_train)

    results = omp_forest.predict(X_train)

    assert isinstance(results, np.ndarray)


def test_regression_omp():

    model_parameters = ModelParameters(
        1, False, ['train+dev', 'train+dev'], False, 1,
        {'n_estimators': 100}, 'omp'
    )

    omp_forest = OmpForestRegressor(model_parameters)
    X_train = [[1, 0], [0, 1]]
    y_train = [-1, 1]

    omp_forest.fit(X_train, y_train, X_train, y_train)

    results = omp_forest.predict(X_train)

    assert isinstance(results, np.ndarray)

def test_multiclassif_omp():

    model_parameters = ModelParameters(
        1, False, ['train+dev', 'train+dev'], False, 1,
        {'n_estimators': 100}, 'omp'
    )

    omp_forest = OmpForestMulticlassClassifier(model_parameters)
    X_train = [[1, 0], [0, 1]]
    y_train = [-1, 1]

    omp_forest.fit(X_train, y_train, X_train, y_train)

    results = omp_forest.predict(X_train)

    assert isinstance(results, np.ndarray)
