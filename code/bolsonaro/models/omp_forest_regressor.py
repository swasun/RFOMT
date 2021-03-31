from bolsonaro.models.omp_forest import SingleOmpForest

from sklearn.ensemble import RandomForestRegressor
import numpy as np


class OmpForestRegressor(SingleOmpForest):

    DEFAULT_SCORE_METRIC = 'mse'

    def __init__(self, models_parameters):
        estimator = RandomForestRegressor(**models_parameters.hyperparameters,
                                          random_state=models_parameters.seed, n_jobs=-1)

        super().__init__(models_parameters, estimator)

    def score_base_estimator(self, X, y):
        predictions = self._base_estimator_predictions(X)
        evaluation = np.mean(np.square(np.mean(predictions, axis=1) - y))
        return evaluation


    def score(self, X, y, metric=DEFAULT_SCORE_METRIC):
        """
        Evaluate OMPForestRegressor on (`X`, `y`) using `metric`

        :param X:
        :param y:
        :param metric:
        :return:
        """
        predictions = self.predict(X)

        if metric == 'mse':
            evaluation = np.mean(np.square(predictions - y))
        else:
            raise ValueError("Unsupported metric '{}'.".format(metric))

        return evaluation
