from bolsonaro import LOG_PATH
from bolsonaro.data.dataset_loader import DatasetLoader
from bolsonaro.data.dataset_parameters import DatasetParameters
from bolsonaro.data.task import Task
from bolsonaro.error_handling.logger_factory import LoggerFactory
from bolsonaro.hyperparameter_searcher import HyperparameterSearcher
from bolsonaro.utils import save_obj_to_json, tqdm_joblib, is_int, is_float

import argparse
import os
import pathlib
import pickle
import random
from dotenv import find_dotenv, load_dotenv
from joblib import Parallel, delayed
from tqdm import tqdm
import threading
import numpy as np
import math
from collections import Counter
from itertools import chain, combinations

"""
I had to install skopt from this repository
https://github.com/darenr/scikit-optimize that handles
the issue described here https://github.com/scikit-optimize/scikit-optimize/issues/762.
"""
from skopt.space import Categorical, Integer


def clean_numpy_int_dict(dictionary):
    return dict([a, int(x)] if type(x) == Integer else
                [a, clean_numpy_int_dict(x)] if type(x) == dict else
                [a, clean_numpy_int_list(x)] if type(x) == list else [a, (x)]
                for a, x in dictionary.items())


def clean_numpy_int_list(list_n):
    return [int(elem) if type(elem) == Integer else
            clean_numpy_int_dict(elem) if type(elem) == dict else
            clean_numpy_int_list(elem) if type(elem) == list else elem
            for elem in list_n]

def process_job(dataset_name, seed, param_space, args):
    logger = LoggerFactory.create(LOG_PATH, 'hyperparameter-searcher_seed{}_ti{}'.format(
        seed, threading.get_ident()))
    logger.info('seed={}'.format(seed))

    dataset = DatasetLoader.load_default(dataset_name, seed)

    if dataset.task == Task.REGRESSION:
        scorer = 'neg_mean_squared_error'
    else:
        scorer = 'accuracy'

    bayesian_searcher = HyperparameterSearcher()
    opt = bayesian_searcher.search(dataset, param_space, args.n_iter,
        args.cv, seed, scorer)

    return {
        '_scorer': scorer,
        '_best_score_train': opt.best_score_,
        '_best_score_test': opt.score(dataset.X_test, dataset.y_test),
        '_best_parameters': clean_numpy_int_dict(opt.best_params_),
        '_random_seed': seed
    }

def run_hyperparameter_search_jobs(seeds, dataset_name, param_space, args):
    # Run one hyperparameter search job per seed
    with tqdm_joblib(tqdm(total=len(seeds), disable=not args.verbose)) as progress_bar:
        opt_results = Parallel(n_jobs=args.job_number)(delayed(process_job)(
            dataset_name, seeds[i], param_space, args) for i in range(len(seeds)))
    return opt_results

def compute_best_params_over_seeds(seeds, dataset_name, param_space, args):
    opt_results = run_hyperparameter_search_jobs(seeds, dataset_name, param_space, args)

    # Move k best_parameters to a list of dict
    all_best_params = [opt_result['_best_parameters'] for opt_result in opt_results]

    """
    list of hyperparam dicts -> list of hyperparam list
    where each element of form 'key:value' becomes 'key_value'
    to afterwards count most common pairs.
    """
    stringify_best_params = list()
    for current_best_params in all_best_params:
        new_best_params = list()
        for key, value in current_best_params.items():
            new_best_params.append(key + '_' + str(value))
        stringify_best_params.append(new_best_params)

    # Compute pair combinations
    pair_combinations = chain.from_iterable(combinations(line, 2) for line in stringify_best_params)

    # Count most common pair combinations in ascent order
    most_common_pair_combinations = Counter(pair_combinations).most_common()

    """
    Select the most frequent hyperparameter values
    until all different hyperparameter variables are
    filled.
    """
    all_param_names = all_best_params[0].keys()
    best_params = dict()
    for pair, _ in most_common_pair_combinations:
        for element in pair:
            split = element.split('_')
            param, value = '_'.join(split[:-1]), split[-1]
            if param not in best_params:
                if is_int(value):
                    value = int(value)
                elif is_float(value):
                    value = float(value)
                best_params[param] = value
        if len(best_params) == len(all_param_names):
            break

    return {
        '_scorer': opt_results[0]['_scorer'],
        '_best_score_train': np.mean([opt_result['_best_score_train'] for opt_result in opt_results]),
        '_best_score_test': np.mean([opt_result['_best_score_test'] for opt_result in opt_results]),
        '_best_parameters': best_params,
        '_random_seed': [opt_result['_random_seed'] for opt_result in opt_results]
    }


if __name__ == "__main__":
    # get environment variables in .env
    load_dotenv(find_dotenv('.env'))

    DEFAULT_CV = 3
    DEFAULT_N_ITER = 50
    DEFAULT_VERBOSE = False
    DEFAULT_JOB_NUMBER = -1
    DICT_PARAM_SPACE = {'n_estimators': Integer(10, 1000),
                        'min_samples_leaf': Integer(1, 1000),
                        'max_depth': Integer(1, 20),
                        'max_features': Categorical(['auto', 'sqrt', 'log2'], [0.5, 0.25, 0.25])}
    begin_random_seed_range = 1
    end_random_seed_range = 2000
    DEFAULT_USE_VARIABLE_SEED_NUMBER = False

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cv', nargs='?', type=int, default=DEFAULT_CV, help='Specify the size of the cross-validation.')
    parser.add_argument('--n_iter', nargs='?', type=int, default=DEFAULT_N_ITER, help='Specify the number of iterations for the bayesian search.')
    parser.add_argument('--random_seed_number', nargs='?', type=int, default=DatasetLoader.DEFAULT_RANDOM_SEED_NUMBER, help='Number of random seeds used.')
    parser.add_argument('--seeds', nargs='+', type=int, default=None, help='Specific a list of seeds instead of generate them randomly')
    parser.add_argument('--use_variable_seed_number', action='store_true', default=DEFAULT_USE_VARIABLE_SEED_NUMBER, help='Compute the amount of random seeds depending on the dataset.')
    parser.add_argument('--datasets', nargs='+', type=str, default=DatasetLoader.dataset_names, help='Specify the dataset used by the estimator.')
    parser.add_argument('--verbose', action='store_true', default=DEFAULT_VERBOSE, help='Print tqdm progress bar.')
    parser.add_argument('--job_number', nargs='?', type=int, default=DEFAULT_JOB_NUMBER, help='Specify the number of job used during the parallelisation across seeds.')
    args = parser.parse_args()

    logger = LoggerFactory.create(LOG_PATH, os.path.basename(__file__))

    if args.seeds != None and args.random_seed_number > 1:
        logger.warning('seeds and random_seed_number parameters are both specified. Seeds will be used.')    

    # Seeds are either provided as parameters or generated at random
    if not args.use_variable_seed_number:
        seeds = args.seeds if args.seeds is not None \
            else [random.randint(begin_random_seed_range, end_random_seed_range) \
            for i in range(args.random_seed_number)]

    for dataset_name in args.datasets:
        dataset_dir = os.path.join('experiments', dataset_name, 'stage1')
        pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)

        logger.info('Bayesian search on dataset {}'.format(dataset_name))
        
        """
        Compute the amount of random seeds as specified in
        DatasetLoader.dataset_seed_numbers dictionary, depending on
        the dataset.
        """
        if args.use_variable_seed_number:
            seeds = [random.randint(begin_random_seed_range, end_random_seed_range) \
                for i in range(DatasetLoader.dataset_seed_numbers[dataset_name])]

        dict_results = compute_best_params_over_seeds(seeds, dataset_name,
            DICT_PARAM_SPACE, args)

        save_obj_to_json(os.path.join(dataset_dir, 'params.json'), dict_results)
