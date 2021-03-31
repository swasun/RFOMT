'''
This module is used to find the best hyperparameters for a given dataset.
'''

from bolsonaro.data.dataset_parameters import DatasetParameters
from bolsonaro.data.dataset_loader import DatasetLoader
from bolsonaro.data.task import Task
from bolsonaro.error_handling.logger_factory import LoggerFactory
from . import LOG_PATH

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from skopt import BayesSearchCV


class HyperparameterSearcher(object):

    def __init__(self):
        self._logger = LoggerFactory.create(LOG_PATH, __name__)

    def search(self, dataset, hyperparameter_space, n_iter, cv,
               random_seed, scorer, verbose=False):
        '''
        For a given dataset and the space of hyperparameters, does a
        bayesian hyperparameters search.
        :input dataset: a Dataset object
        :input hyperparameter_space: a dictionnary, keys are hyperparameters,
        value their spaces defined with skopt
        :input n_iter: the number of iterations of the bayesian search
        :input cv: the size of the cross validation
        :input random_seed: int, the seed for the bayesian search
        :input scorer: str, the name of the scorer
        :input verbose: bool, print state of the research
        :return: a skopt.searchcv.BayesSearchCV object
        '''

        if dataset.task == Task.REGRESSION:
            estimator = RandomForestRegressor(n_jobs=-1, random_state=random_seed)
        else:
            estimator = RandomForestClassifier(n_jobs=-1, random_state=random_seed)

        opt = BayesSearchCV(estimator, hyperparameter_space, n_iter=n_iter,
                            cv=cv, n_jobs=-1, random_state=random_seed,
                            scoring=scorer, verbose=verbose)

        opt.fit(dataset.X_train, dataset.y_train)

        return opt
