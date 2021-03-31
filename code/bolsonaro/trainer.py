from bolsonaro.models.model_raw_results import ModelRawResults
from bolsonaro.models.omp_forest_regressor import OmpForestRegressor
from bolsonaro.models.omp_forest_classifier import OmpForestBinaryClassifier, OmpForestMulticlassClassifier
from bolsonaro.models.nn_omp_forest_regressor import NonNegativeOmpForestRegressor
from bolsonaro.models.nn_omp_forest_classifier import NonNegativeOmpForestBinaryClassifier
from bolsonaro.models.similarity_forest_regressor import SimilarityForestRegressor, SimilarityForestClassifier
from bolsonaro.models.kmeans_forest_regressor import KMeansForestRegressor, KMeansForestClassifier
from bolsonaro.models.ensemble_selection_forest_regressor import EnsembleSelectionForestRegressor, EnsembleSelectionForestClassifier
from bolsonaro.error_handling.logger_factory import LoggerFactory
from bolsonaro.data.task import Task
from . import LOG_PATH

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import normalize
import time
import datetime
import numpy as np
import os
import pickle


class Trainer(object):
    """
    Class capable of fitting any model object to some prepared data then evaluate and save results through the `train` method.
    """

    def __init__(self, dataset, regression_score_metric=mean_squared_error, classification_score_metric=accuracy_score,
        base_regression_score_metric=mean_squared_error, base_classification_score_metric=accuracy_score):
        """

        :param dataset: Object with X_train, y_train, X_dev, y_dev, X_test and Y_test attributes
        """
        self._dataset = dataset
        self._logger = LoggerFactory.create(LOG_PATH, __name__)
        self._regression_score_metric = regression_score_metric
        self._classification_score_metric = classification_score_metric
        self._base_regression_score_metric = base_regression_score_metric
        self._base_classification_score_metric = base_classification_score_metric
        self._score_metric_name = regression_score_metric.__name__ if dataset.task == Task.REGRESSION \
            else classification_score_metric.__name__
        self._base_score_metric_name = base_regression_score_metric.__name__ if dataset.task == Task.REGRESSION \
            else base_classification_score_metric.__name__

    @property
    def score_metric_name(self):
        return self._score_metric_name

    @property
    def base_score_metric_name(self):
        return self._base_score_metric_name

    def init(self, model, subsets_used='train,dev'):
        if type(model) in [RandomForestRegressor, RandomForestClassifier]:
            if subsets_used == 'train,dev':
                self._X_forest = self._dataset.X_train
                self._y_forest = self._dataset.y_train
            else:
                self._X_forest = np.concatenate([self._dataset.X_train, self._dataset.X_dev])
                self._y_forest = np.concatenate([self._dataset.y_train, self._dataset.y_dev])    
            self._logger.debug('Fitting the forest on train subset')
        elif model.models_parameters.subsets_used == 'train,dev':
            self._X_forest = self._dataset.X_train
            self._y_forest = self._dataset.y_train
            self._X_omp = self._dataset.X_dev
            self._y_omp = self._dataset.y_dev
            self._logger.debug('Fitting the forest on train subset and OMP on dev subset.')
        elif model.models_parameters.subsets_used == 'train+dev,train+dev':
            self._X_forest = np.concatenate([self._dataset.X_train, self._dataset.X_dev])
            self._X_omp = self._X_forest
            self._y_forest = np.concatenate([self._dataset.y_train, self._dataset.y_dev])
            self._y_omp = self._y_forest
            self._logger.debug('Fitting both the forest and OMP on train+dev subsets.')
        elif model.models_parameters.subsets_used == 'train,train+dev':
            self._X_forest = self._dataset.X_train
            self._y_forest = self._dataset.y_train
            self._X_omp = np.concatenate([self._dataset.X_train, self._dataset.X_dev])
            self._y_omp = np.concatenate([self._dataset.y_train, self._dataset.y_dev])
        else:
            raise ValueError("Unknown specified subsets_used parameter '{}'".format(model.models_parameters.subsets_used))

    def train(self, model, extracted_forest_size=None, seed=None, use_distillation=False):
        """
        :param model: An instance of either RandomForestRegressor, RandomForestClassifier, OmpForestRegressor,
            OmpForestBinaryClassifier, OmpForestMulticlassClassifier.
        :return:
        """
        self._logger.debug('Training model using train set...')
        self._begin_time = time.time()
        if type(model) in [RandomForestRegressor, RandomForestClassifier]:
            if extracted_forest_size is not None:
                estimators_index = np.arange(len(model.estimators_))
                np.random.seed(seed)
                np.random.shuffle(estimators_index)
                choosen_estimators = estimators_index[:extracted_forest_size]
                model.estimators_ = np.array(model.estimators_)[choosen_estimators]
            else:
                model.fit(
                    X=self._X_forest,
                    y=self._y_forest
                )
        else:
            if type(model) in [OmpForestRegressor, OmpForestBinaryClassifier, OmpForestMulticlassClassifier,
                NonNegativeOmpForestRegressor, NonNegativeOmpForestBinaryClassifier] and \
                use_distillation:
                model.fit(
                    self._X_forest, # X_train or X_train+X_dev
                    self._y_forest,
                    self._X_omp, # X_train+X_dev or X_dev
                    self._y_omp,
                    use_distillation=use_distillation
                )
            else:
                model.fit(
                    self._X_forest, # X_train or X_train+X_dev
                    self._y_forest,
                    self._X_omp, # X_train+X_dev or X_dev
                    self._y_omp
                )
        self._end_time = time.time()

    def __score_func(self, model, X, y_true, weights=True, extracted_forest_size=None):
        if type(model) in [OmpForestRegressor, RandomForestRegressor]:
            if weights:
                y_pred = model.predict(X)
            else:
                y_pred = model.predict_no_weights(X)
            result = self._regression_score_metric(y_true, y_pred)
        elif type(model) == NonNegativeOmpForestRegressor:
            if weights:
                y_pred = model.predict(X, extracted_forest_size)
            else:
                y_pred = model.predict_no_weights(X, extracted_forest_size)
            result = self._regression_score_metric(y_true, y_pred)
        elif type(model) == NonNegativeOmpForestBinaryClassifier:
            if weights:
                y_pred = model.predict(X, extracted_forest_size)
            else:
                y_pred = model.predict_no_weights(X, extracted_forest_size)
            y_pred = np.sign(y_pred)
            y_pred = np.where(y_pred == 0, 1, y_pred)
            result = self._classification_score_metric(y_true, y_pred)
        elif type(model) in [OmpForestBinaryClassifier, OmpForestMulticlassClassifier, RandomForestClassifier]:
            if weights:
                y_pred = model.predict(X)
            else:
                y_pred = model.predict_no_weights(X)
            if type(model) is OmpForestBinaryClassifier:
                y_pred = np.sign(y_pred)
                y_pred = np.where(y_pred == 0, 1, y_pred)
            result = self._classification_score_metric(y_true, y_pred)
        elif type(model) in [SimilarityForestRegressor, SimilarityForestClassifier, KMeansForestRegressor, EnsembleSelectionForestRegressor, KMeansForestClassifier,
            EnsembleSelectionForestClassifier]:
            result = model.score(X, y_true)
        return result

    def __score_func_base(self, model, X, y_true):
        if type(model) in [OmpForestRegressor, SimilarityForestRegressor, KMeansForestRegressor, EnsembleSelectionForestRegressor,
            NonNegativeOmpForestRegressor]:
            y_pred = model.predict_base_estimator(X)
            result = self._base_regression_score_metric(y_true, y_pred)
        elif type(model) in [OmpForestBinaryClassifier, OmpForestMulticlassClassifier, KMeansForestClassifier,
            SimilarityForestClassifier, EnsembleSelectionForestClassifier, NonNegativeOmpForestBinaryClassifier]:
            y_pred = model.predict_base_estimator(X)
            result = self._base_classification_score_metric(y_true, y_pred)
        elif type(model) == RandomForestClassifier:
            y_pred = model.predict(X)
            result = self._base_classification_score_metric(y_true, y_pred)
        elif type(model) is RandomForestRegressor:
            y_pred = model.predict(X)
            result = self._base_regression_score_metric(y_true, y_pred)
        return result

    def _evaluate_predictions(self, predictions, aggregation_function):
        predictions = normalize(predictions)

        return aggregation_function(np.abs((predictions @ predictions.T - np.eye(len(predictions)))))

    def _compute_forest_strength(self, predictions, y, metric_function):
        scores = np.array([metric_function(y, prediction) for prediction in predictions])
        return scores, np.mean(scores)

    def compute_results(self, model, models_dir, subsets_used='train+dev,train+dev', extracted_forest_size=None):
        """
        :param model: Object with
        :param models_dir: Where the results will be saved
        """
        # Reeeally dirty to put that here but otherwise it's not thread safe...
        if type(model) in [RandomForestRegressor, RandomForestClassifier]:
            if subsets_used == 'train,dev':
                X_forest = self._dataset.X_train
                y_forest = self._dataset.y_train
            else:
                X_forest = np.concatenate([self._dataset.X_train, self._dataset.X_dev])
                y_forest = np.concatenate([self._dataset.y_train, self._dataset.y_dev])
            X_omp = self._dataset.X_dev
            y_omp = self._dataset.y_dev
        elif model.models_parameters.subsets_used == 'train,dev':
            X_forest = self._dataset.X_train
            y_forest = self._dataset.y_train
            X_omp = self._dataset.X_dev
            y_omp = self._dataset.y_dev
        elif model.models_parameters.subsets_used == 'train+dev,train+dev':
            X_forest = np.concatenate([self._dataset.X_train, self._dataset.X_dev])
            X_omp = X_forest
            y_forest = np.concatenate([self._dataset.y_train, self._dataset.y_dev])
            y_omp = y_forest
        elif model.models_parameters.subsets_used == 'train,train+dev':
            X_forest = self._dataset.X_train
            y_forest = self._dataset.y_train
            X_omp = np.concatenate([self._dataset.X_train, self._dataset.X_dev])
            y_omp = np.concatenate([self._dataset.y_train, self._dataset.y_dev])
        else:
            raise ValueError("Unknown specified subsets_used parameter '{}'".format(model.models_parameters.subsets_used))

        model_weights = ''
        if type(model) in [OmpForestRegressor, OmpForestBinaryClassifier]:
            model_weights = model._omp.coef_
        elif type(model) == OmpForestMulticlassClassifier:
            model_weights = model._dct_class_omp
        elif type(model) == OmpForestBinaryClassifier:
            model_weights = model._omp
        elif type(model) in [NonNegativeOmpForestRegressor, NonNegativeOmpForestBinaryClassifier]:
            model_weights = model._omp.get_coef(extracted_forest_size)

        if type(model) in [SimilarityForestRegressor, KMeansForestRegressor, EnsembleSelectionForestRegressor, 
            SimilarityForestClassifier, KMeansForestClassifier, EnsembleSelectionForestClassifier]:
            selected_trees = model.selected_trees
        elif type(model) in [OmpForestRegressor, OmpForestMulticlassClassifier, OmpForestBinaryClassifier,
            NonNegativeOmpForestRegressor, NonNegativeOmpForestBinaryClassifier]:
            selected_trees = np.asarray(model.forest)[model_weights != 0]
        elif type(model) in [RandomForestRegressor, RandomForestClassifier]:
            selected_trees = model.estimators_

        if len(selected_trees) > 0:
            target_selected_tree = int(os.path.split(models_dir)[-1])
            if target_selected_tree != len(selected_trees):
                predictions_X_omp = model.predict(X_omp, extracted_forest_size) \
                    if type(model) in [NonNegativeOmpForestBinaryClassifier, NonNegativeOmpForestRegressor] \
                    else model.predict(X_omp)
                error_prediction = np.linalg.norm(predictions_X_omp - y_omp)
                if not np.isclose(error_prediction, 0):
                    #raise ValueError(f'Invalid selected tree number target_selected_tree:{target_selected_tree} - len(selected_trees):{len(selected_trees)}')
                    self._logger.error(f'Invalid selected tree number target_selected_tree:{target_selected_tree} - len(selected_trees):{len(selected_trees)}')
                else:
                    self._logger.warning(f"Invalid selected tree number target_selected_tree:{target_selected_tree} - len(selected_trees):{len(selected_trees)}"
                                         " But the prediction is perfect on X_omp. Keep less trees.")
            with open(os.path.join(models_dir, 'selected_trees.pickle'), 'wb') as output_file:
                pickle.dump(selected_trees, output_file)

        strength_metric = self._regression_score_metric if self._dataset.task == Task.REGRESSION \
            else lambda y_true, y_pred: self._classification_score_metric(y_true, (y_pred -0.5)*2)

        train_predictions = np.array([tree.predict(X_forest) for tree in selected_trees])
        dev_predictions = np.array([tree.predict(X_omp) for tree in selected_trees])
        test_predictions = np.array([tree.predict(self._dataset.X_test) for tree in selected_trees])

        train_scores, train_strength = self._compute_forest_strength(train_predictions, y_forest, strength_metric)
        dev_scores, dev_strength = self._compute_forest_strength(dev_predictions, y_omp, strength_metric)
        test_scores, test_strength = self._compute_forest_strength(test_predictions, self._dataset.y_test, strength_metric)

        results = ModelRawResults(
            model_weights=model_weights,
            training_time=self._end_time - self._begin_time,
            datetime=datetime.datetime.now(),
            train_score=self.__score_func(model, X_forest, y_forest, extracted_forest_size=extracted_forest_size),
            dev_score=self.__score_func(model, X_omp, y_omp, extracted_forest_size=extracted_forest_size),
            test_score=self.__score_func(model, self._dataset.X_test, self._dataset.y_test, extracted_forest_size=extracted_forest_size),
            train_score_base=self.__score_func_base(model, X_forest, y_forest),
            dev_score_base=self.__score_func_base(model, X_omp, y_omp),
            test_score_base=self.__score_func_base(model, self._dataset.X_test, self._dataset.y_test),
            score_metric=self._score_metric_name,
            base_score_metric=self._base_score_metric_name,
            train_coherence=self._evaluate_predictions(train_predictions, aggregation_function=np.max),
            dev_coherence=self._evaluate_predictions(dev_predictions, aggregation_function=np.max),
            test_coherence=self._evaluate_predictions(test_predictions, aggregation_function=np.max),
            train_correlation=self._evaluate_predictions(train_predictions, aggregation_function=np.mean),
            dev_correlation=self._evaluate_predictions(dev_predictions, aggregation_function=np.mean),
            test_correlation=self._evaluate_predictions(test_predictions, aggregation_function=np.mean),
            train_scores=train_scores,
            dev_scores=dev_scores,
            test_scores=test_scores,
            train_strength=train_strength,
            dev_strength=dev_strength,
            test_strength=test_strength
        )
        results.save(models_dir)
        self._logger.info("Base performance on test: {}".format(results.test_score_base))
        self._logger.info("Performance on test: {}".format(results.test_score))

        self._logger.info("Base performance on train: {}".format(results.train_score_base))
        self._logger.info("Performance on train: {}".format(results.train_score))

        self._logger.info("Base performance on dev: {}".format(results.dev_score_base))
        self._logger.info("Performance on dev: {}".format(results.dev_score))

        self._logger.info(f'test_coherence: {results.test_coherence}')
        self._logger.info(f'test_correlation: {results.test_correlation}')
        self._logger.info(f'test_strength: {results.test_strength}')

        if type(model) in [OmpForestBinaryClassifier, OmpForestRegressor, OmpForestMulticlassClassifier,
            NonNegativeOmpForestBinaryClassifier, NonNegativeOmpForestRegressor]:
            results = ModelRawResults(
                model_weights='',
                training_time=self._end_time - self._begin_time,
                datetime=datetime.datetime.now(),
                train_score=self.__score_func(model, X_forest, y_forest, False, extracted_forest_size=extracted_forest_size),
                dev_score=self.__score_func(model, X_omp, y_omp, False, extracted_forest_size=extracted_forest_size),
                test_score=self.__score_func(model, self._dataset.X_test, self._dataset.y_test, False, extracted_forest_size=extracted_forest_size),
                train_score_base=self.__score_func_base(model, X_forest, y_forest),
                dev_score_base=self.__score_func_base(model, X_omp, y_omp),
                test_score_base=self.__score_func_base(model, self._dataset.X_test, self._dataset.y_test),
                score_metric=self._score_metric_name,
                base_score_metric=self._base_score_metric_name,
                train_scores=train_scores,
                dev_scores=dev_scores,
                test_scores=test_scores
            )
            results.save(models_dir+'_no_weights')
            self._logger.info("Base performance on test without weights: {}".format(results.test_score_base))
            self._logger.info("Performance on test without weights: {}".format(results.test_score))

            self._logger.info("Base performance on train without weights: {}".format(results.train_score_base))
            self._logger.info("Performance on train without weights: {}".format(results.train_score))

            self._logger.info("Base performance on dev without weights: {}".format(results.dev_score_base))
            self._logger.info("Performance on dev without weights: {}".format(results.dev_score))
