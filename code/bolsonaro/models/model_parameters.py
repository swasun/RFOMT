from bolsonaro.utils import save_obj_to_json, load_obj_from_json

import os


class ModelParameters(object):

    def __init__(self, extracted_forest_size, normalize_D, subsets_used,
        normalize_weights, seed, hyperparameters, extraction_strategy, intermediate_solutions_sizes=None):
        """Init of ModelParameters.
        
        Args:
            extracted_forest_size (int): extracted forest size
            intermediate_solutions_sizes (list): list of all intermediate solutions sizes
            normalize_D (bool): true normalize the distribution, false no
            subsets_used (list): which dataset use for randomForest and for OMP
                'train', 'dev' or 'train+dev' and combination of two of this.
            normalize_weights (bool): if we normalize the weights or no.
            seed (int): the seed used for the randomization.
            hyperparameters (dict): dict of the hyperparameters of RandomForest
                in scikit-learn.
            extraction_strategy (str): either 'none', 'random', 'omp'
        """
        self._extracted_forest_size = extracted_forest_size
        self._normalize_D = normalize_D
        self._subsets_used = subsets_used
        self._normalize_weights = normalize_weights
        self._seed = seed
        self._hyperparameters = hyperparameters
        self._extraction_strategy = extraction_strategy

        if self._extraction_strategy == 'omp_nn' and intermediate_solutions_sizes is None:
            raise ValueError("Intermediate solutions must be set if non negative option is on.")
        self._intermediate_solutions_sizes = intermediate_solutions_sizes

    @property
    def intermediate_solutions_sizes(self):
        return self._intermediate_solutions_sizes

    @property
    def extracted_forest_size(self):
        return self._extracted_forest_size

    @property
    def normalize_D(self):
        return self._normalize_D

    @property
    def subsets_used(self):
        return self._subsets_used

    @property
    def normalize_weights(self):
        return self._normalize_weights

    @property
    def seed(self):
        return self._seed

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @property
    def extraction_strategy(self):
        return self._extraction_strategy

    def save(self, directory_path, experiment_id):
        save_obj_to_json(directory_path + os.sep + 'model_parameters_{}.json'.format(experiment_id),
            self.__dict__)

    @staticmethod
    def load(directory_path, experiment_id):
        return load_obj_from_json(directory_path + os.sep + 'model_parameters_{}.json'.format(experiment_id),
            ModelParameters)
