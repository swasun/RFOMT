from bolsonaro.models.omp_forest_classifier import OmpForestBinaryClassifier, OmpForestMulticlassClassifier
from bolsonaro.models.omp_forest_regressor import OmpForestRegressor
from bolsonaro.models.nn_omp_forest_regressor import NonNegativeOmpForestRegressor
from bolsonaro.models.nn_omp_forest_classifier import NonNegativeOmpForestBinaryClassifier
from bolsonaro.models.model_parameters import ModelParameters
from bolsonaro.models.similarity_forest_regressor import SimilarityForestRegressor, SimilarityForestClassifier
from bolsonaro.models.kmeans_forest_regressor import KMeansForestRegressor, KMeansForestClassifier
from bolsonaro.models.ensemble_selection_forest_regressor import EnsembleSelectionForestRegressor, EnsembleSelectionForestClassifier
from bolsonaro.data.task import Task

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import os
import pickle


class ModelFactory(object):

    @staticmethod
    def build(task, model_parameters):
        if task not in [Task.BINARYCLASSIFICATION, Task.REGRESSION, Task.MULTICLASSIFICATION]:
            raise ValueError("Unsupported task '{}'".format(task))

        if task == Task.BINARYCLASSIFICATION:
            if model_parameters.extraction_strategy in ['omp', 'omp_distillation']:
                return OmpForestBinaryClassifier(model_parameters)
            elif model_parameters.extraction_strategy == 'omp_nn':
                return NonNegativeOmpForestBinaryClassifier(model_parameters)
            elif model_parameters.extraction_strategy == 'random':
                return RandomForestClassifier(**model_parameters.hyperparameters,
                    random_state=model_parameters.seed)
            elif model_parameters.extraction_strategy == 'none':
                return RandomForestClassifier(**model_parameters.hyperparameters,
                    random_state=model_parameters.seed)
            elif model_parameters.extraction_strategy == 'ensemble':
                return EnsembleSelectionForestClassifier(model_parameters)
            elif model_parameters.extraction_strategy == 'kmeans':
                return KMeansForestClassifier(model_parameters)
            elif model_parameters.extraction_strategy in ['similarity_similarities', 'similarity_predictions']:
                return SimilarityForestClassifier(model_parameters)
            else:
                raise ValueError('Invalid extraction strategy')
        elif task == Task.REGRESSION:
            if model_parameters.extraction_strategy in ['omp', 'omp_distillation']:
                return OmpForestRegressor(model_parameters)
            elif model_parameters.extraction_strategy == 'omp_nn':
                return NonNegativeOmpForestRegressor(model_parameters)
            elif model_parameters.extraction_strategy == 'random':
                return RandomForestRegressor(**model_parameters.hyperparameters,
                    random_state=model_parameters.seed)
            elif model_parameters.extraction_strategy in ['similarity_similarities', 'similarity_predictions']:
                return SimilarityForestRegressor(model_parameters)
            elif model_parameters.extraction_strategy == 'kmeans':
                return KMeansForestRegressor(model_parameters)
            elif model_parameters.extraction_strategy == 'ensemble':
                return EnsembleSelectionForestRegressor(model_parameters)
            elif model_parameters.extraction_strategy == 'none':
                return RandomForestRegressor(**model_parameters.hyperparameters,
                    random_state=model_parameters.seed)
            else:
                raise ValueError('Invalid extraction strategy')
        elif task == Task.MULTICLASSIFICATION:
            if model_parameters.extraction_strategy in ['omp', 'omp_distillation']:
                return OmpForestMulticlassClassifier(model_parameters)
            elif model_parameters.extraction_strategy == 'omp_nn':
                raise ValueError('omp_nn is unsuported for multi classification')
            elif model_parameters.extraction_strategy == 'random':
                return RandomForestClassifier(**model_parameters.hyperparameters,
                    random_state=model_parameters.seed)
            elif model_parameters.extraction_strategy == 'none':
                return RandomForestClassifier(**model_parameters.hyperparameters,
                    random_state=model_parameters.seed)
            else:
                raise ValueError('Invalid extraction strategy')
