from bolsonaro.data.dataset_parameters import DatasetParameters
from bolsonaro.data.dataset_loader import DatasetLoader
from bolsonaro.models.model_factory import ModelFactory
from bolsonaro.models.model_parameters import ModelParameters
from bolsonaro.models.ensemble_selection_forest_regressor import EnsembleSelectionForestRegressor
from bolsonaro.trainer import Trainer
from bolsonaro.utils import resolve_experiment_id, tqdm_joblib
from bolsonaro import LOG_PATH
from bolsonaro.error_handling.logger_factory import LoggerFactory

from dotenv import find_dotenv, load_dotenv
import argparse
import copy
import json
import pathlib
import random
import os
from joblib import Parallel, delayed
import threading
import json
from tqdm import tqdm
import numpy as np
import shutil


def seed_job(seed_job_pb, seed, parameters, experiment_id, hyperparameters, verbose):
    """
    Experiment function.

    Will be used as base function for worker in multithreaded application.

    :param seed:
    :param parameters:
    :param experiment_id:
    :return:
    """
    logger = LoggerFactory.create(LOG_PATH, 'training_seed{}_ti{}'.format(
        seed, threading.get_ident()))

    seed_str = str(seed)
    experiment_id_str = str(experiment_id)
    models_dir = parameters['models_dir'] + os.sep + experiment_id_str + os.sep + 'seeds' + \
        os.sep + seed_str
    pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)

    dataset_parameters = DatasetParameters(
        name=parameters['dataset_name'],
        test_size=parameters['test_size'],
        dev_size=parameters['dev_size'],
        random_state=seed,
        dataset_normalizer=parameters['dataset_normalizer']
    )
    dataset_parameters.save(models_dir, experiment_id_str)
    dataset = DatasetLoader.load(dataset_parameters)

    trainer = Trainer(dataset)

    if parameters['extraction_strategy'] == 'random':
        pretrained_model_parameters = ModelParameters(
            extracted_forest_size=parameters['forest_size'],
            normalize_D=parameters['normalize_D'],
            subsets_used=parameters['subsets_used'],
            normalize_weights=parameters['normalize_weights'],
            seed=seed,
            hyperparameters=hyperparameters,
            extraction_strategy=parameters['extraction_strategy']
        )
        pretrained_estimator = ModelFactory.build(dataset.task, pretrained_model_parameters)
        pretrained_trainer = Trainer(dataset)
        pretrained_trainer.init(pretrained_estimator, subsets_used=parameters['subsets_used'])
        pretrained_estimator.fit(
            X=pretrained_trainer._X_forest,
            y=pretrained_trainer._y_forest
        )
    else:
        pretrained_estimator = None
        pretrained_model_parameters = None

    if parameters['extraction_strategy'] == 'none':
        forest_size = hyperparameters['n_estimators']
        logger.info('Base forest training with fixed forest size of {}'.format(forest_size))
        sub_models_dir = models_dir + os.sep + 'forest_size' + os.sep + str(forest_size)

        # Check if the result file already exists
        already_exists = False
        if os.path.isdir(sub_models_dir):
            sub_models_dir_files = os.listdir(sub_models_dir)
            for file_name in sub_models_dir_files:
                if file_name == 'model_raw_results.pickle':
                    already_exists = os.path.getsize(os.path.join(sub_models_dir, file_name)) > 0
                    break
                else:
                    continue
        if already_exists:
            logger.info('Base forest result already exists. Skipping...')
        else:
            pathlib.Path(sub_models_dir).mkdir(parents=True, exist_ok=True)
            model_parameters = ModelParameters(
                extracted_forest_size=forest_size,
                normalize_D=parameters['normalize_D'],
                subsets_used=parameters['subsets_used'],
                normalize_weights=parameters['normalize_weights'],
                seed=seed,
                hyperparameters=hyperparameters,
                extraction_strategy=parameters['extraction_strategy']
            )
            model_parameters.save(sub_models_dir, experiment_id)

            model = ModelFactory.build(dataset.task, model_parameters)

            trainer.init(model, subsets_used=parameters['subsets_used'])
            trainer.train(model)
            trainer.compute_results(model, sub_models_dir)
    elif parameters['extraction_strategy'] == 'omp_nn':
        forest_size = hyperparameters['n_estimators']
        model_parameters = ModelParameters(
            extracted_forest_size=forest_size,
            normalize_D=parameters['normalize_D'],
            subsets_used=parameters['subsets_used'],
            normalize_weights=parameters['normalize_weights'],
            seed=seed,
            hyperparameters=hyperparameters,
            extraction_strategy=parameters['extraction_strategy'],
            intermediate_solutions_sizes=parameters['extracted_forest_size']
        )

        model = ModelFactory.build(dataset.task, model_parameters)

        trainer.init(model, subsets_used=parameters['subsets_used'])
        trainer.train(model)
        for extracted_forest_size in parameters['extracted_forest_size']:
            sub_models_dir = models_dir + os.sep + 'extracted_forest_sizes' + os.sep + str(extracted_forest_size)
            pathlib.Path(sub_models_dir).mkdir(parents=True, exist_ok=True)
            trainer.compute_results(model, sub_models_dir, extracted_forest_size=extracted_forest_size)
            model_parameters.save(sub_models_dir, experiment_id)
    else:
        with tqdm_joblib(tqdm(total=len(parameters['extracted_forest_size']), disable=not verbose)) as extracted_forest_size_job_pb:
            Parallel(n_jobs=-1)(delayed(extracted_forest_size_job)(extracted_forest_size_job_pb, parameters['extracted_forest_size'][i],
                models_dir, seed, parameters, dataset, hyperparameters, experiment_id, trainer,
                pretrained_estimator=pretrained_estimator, pretrained_model_parameters=pretrained_model_parameters,
                use_distillation=parameters['extraction_strategy'] == 'omp_distillation')
                for i in range(len(parameters['extracted_forest_size'])))

    logger.info(f'Training done for seed {seed_str}')
    seed_job_pb.update(1)

def extracted_forest_size_job(extracted_forest_size_job_pb, extracted_forest_size, models_dir,
    seed, parameters, dataset, hyperparameters, experiment_id, trainer,
    pretrained_estimator=None, pretrained_model_parameters=None, use_distillation=False):

    logger = LoggerFactory.create(LOG_PATH, 'training_seed{}_extracted_forest_size{}_ti{}'.format(
        seed, extracted_forest_size, threading.get_ident()))
    logger.info('extracted_forest_size={}'.format(extracted_forest_size))

    sub_models_dir = models_dir + os.sep + 'extracted_forest_sizes' + os.sep + str(extracted_forest_size)

    # Check if the result file already exists
    already_exists = False
    if os.path.isdir(sub_models_dir):
        sub_models_dir_files = os.listdir(sub_models_dir)
        for file_name in sub_models_dir_files:
            if file_name == 'model_raw_results.pickle':
                already_exists = os.path.getsize(os.path.join(sub_models_dir, file_name)) > 0
                break
            else:
                continue
    if already_exists:
        logger.info(f'Extracted forest {extracted_forest_size} result already exists. Skipping...')
        return

    pathlib.Path(sub_models_dir).mkdir(parents=True, exist_ok=True)

    if not pretrained_estimator:
        model_parameters = ModelParameters(
            extracted_forest_size=extracted_forest_size,
            normalize_D=parameters['normalize_D'],
            subsets_used=parameters['subsets_used'],
            normalize_weights=parameters['normalize_weights'],
            seed=seed,
            hyperparameters=hyperparameters,
            extraction_strategy=parameters['extraction_strategy']
        )
        model_parameters.save(sub_models_dir, experiment_id)
        model = ModelFactory.build(dataset.task, model_parameters)
    else:
        model = copy.deepcopy(pretrained_estimator)
        pretrained_model_parameters.save(sub_models_dir, experiment_id)

    trainer.init(model, subsets_used=parameters['subsets_used'])
    trainer.train(model, extracted_forest_size=extracted_forest_size, seed=seed,
        use_distillation=use_distillation)
    trainer.compute_results(model, sub_models_dir)

"""
Command lines example for stage 1:
python code/train.py --dataset_name=california_housing --seeds 1 2 3 4 5 --extraction_strategy=none --save_experiment_configuration 1 none_with_params --extracted_forest_size_stop=0.05
python code/train.py --dataset_name=california_housing --seeds 1 2 3 4 5 --extraction_strategy=random --save_experiment_configuration 1 random_with_params --extracted_forest_size_stop=0.05
python code/train.py --dataset_name=california_housing --seeds 1 2 3 4 5 --save_experiment_configuration 1 omp_with_params --extracted_forest_size_stop=0.05
python code/train.py --dataset_name=california_housing --seeds 1 2 3 4 5 --extraction_strategy=none --skip_best_hyperparams --save_experiment_configuration 1 none_wo_params --extracted_forest_size_stop=0.05
python code/train.py --dataset_name=california_housing --seeds 1 2 3 4 5 --extraction_strategy=random --skip_best_hyperparams --save_experiment_configuration 1 random_wo_params --extracted_forest_size_stop=0.05
python code/train.py --dataset_name=california_housing --seeds 1 2 3 4 5 --skip_best_hyperparams --save_experiment_configuration 1 omp_wo_params --extracted_forest_size_stop=0.05
python code/compute_results.py --stage 1 --experiment_ids 1 2 3 4 5 6 --dataset_name=california_housing

Command lines example for stage 2:
python code/train.py --dataset_name=california_housing --seeds 1 2 3 4 5 --save_experiment_configuration 2 no_normalization --extracted_forest_size_stop=0.05
python code/train.py --dataset_name=california_housing --seeds 1 2 3 4 5 --save_experiment_configuration 2 normalize_D --normalize_D --extracted_forest_size_stop=0.05
python code/train.py --dataset_name=california_housing --seeds 1 2 3 4 5 --save_experiment_configuration 2 normalize_weights --normalize_weights --extracted_forest_size_stop=0.05
python code/train.py --dataset_name=california_housing --seeds 1 2 3 4 5 --save_experiment_configuration 2 normalize_D_and_weights --normalize_D --normalize_weights --extracted_forest_size_stop=0.05
python code/compute_results.py --stage 2 --experiment_ids 7 8 9 10 --dataset_name=california_housing

Command lines example for stage 3:
python code/train.py --dataset_name=california_housing --seeds 1 2 3 4 5 --save_experiment_configuration 3 train-dev_subset --extracted_forest_size_stop=0.05 --subsets_used train,dev
python code/train.py --dataset_name=california_housing --seeds 1 2 3 4 5 --save_experiment_configuration 3 train-dev_train-dev_subset --extracted_forest_size_stop=0.05 --subsets_used train+dev,train+dev
python code/train.py --dataset_name=california_housing --seeds 1 2 3 4 5 --save_experiment_configuration 3 train-train-dev_subset --extracted_forest_size_stop=0.05 --subsets_used train,train+dev
python code/compute_results.py --stage 3 --experiment_ids 11 12 13 --dataset_name=california_housing

Command lines example for stage 4:
python code/train.py --dataset_name=california_housing --seeds 1 2 3 4 5 --extraction_strategy=none --save_experiment_configuration 4 none_with_params --extracted_forest_size_stop=0.05
python code/train.py --dataset_name=california_housing --seeds 1 2 3 4 5 --extraction_strategy=random --save_experiment_configuration 4 random_with_params --extracted_forest_size_stop=0.05
python code/train.py --dataset_name=california_housing --seeds 1 2 3 4 5 --save_experiment_configuration 4 omp_with_params --extracted_forest_size_stop=0.05 --subsets_used train+dev,train+dev
python code/compute_results.py --stage 4 --experiment_ids 1 2 3 --dataset_name=california_housing
"""
if __name__ == "__main__":
    load_dotenv(find_dotenv('.env'))
    DEFAULT_EXPERIMENT_CONFIGURATION_PATH = 'experiments'
    # the models will be stored in a directory structure like: models/{experiment_id}/seeds/{seed_nb}/extracted_forest_sizes/{extracted_forest_size}
    DEFAULT_MODELS_DIR = os.environ['project_dir'] + os.sep + 'models'
    DEFAULT_VERBOSE = False
    DEFAULT_SKIP_BEST_HYPERPARAMS = False
    DEFAULT_JOB_NUMBER = -1
    DEFAULT_EXTRACTION_STRATEGY = 'omp'
    DEFAULT_OVERWRITE = False

    begin_random_seed_range = 1
    end_random_seed_range = 2000

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment_id', nargs='?', type=int, default=None, help='Specify an experiment id. Remove already existing model with this specified experiment id.')
    parser.add_argument('--experiment_configuration', nargs='?', type=str, default=None, help='Specify an experiment configuration file name. Overload all other parameters.')
    parser.add_argument('--experiment_configuration_path', nargs='?', type=str, default=DEFAULT_EXPERIMENT_CONFIGURATION_PATH, help='Specify the experiment configuration directory path.')
    parser.add_argument('--dataset_name', nargs='?', type=str, default=DatasetLoader.DEFAULT_DATASET_NAME, help='Specify the dataset. Regression: boston, diabetes, linnerud, california_housing. Classification: iris, digits, wine, breast_cancer, olivetti_faces, 20newsgroups, 20newsgroups_vectorized, lfw_people, lfw_pairs, covtype, rcv1, kddcup99.')
    parser.add_argument('--normalize_D', action='store_true', default=DatasetLoader.DEFAULT_NORMALIZE_D, help='Specify if we want to normalize the prediction of the forest by doing the L2 division of the pred vectors.')
    parser.add_argument('--dataset_normalizer', nargs='?', type=str, default=DatasetLoader.DEFAULT_DATASET_NORMALIZER, help='Specify which dataset normalizer use (either standard, minmax, robust or normalizer).')
    parser.add_argument('--forest_size', nargs='?', type=int, default=None, help='The number of trees of the random forest.')
    parser.add_argument('--extracted_forest_size_samples', nargs='?', type=int, default=DatasetLoader.DEFAULT_EXTRACTED_FOREST_SIZE_SAMPLES, help='The number of extracted forest sizes (proportional to the forest size) selected by OMP.')
    parser.add_argument('--extracted_forest_size_stop', nargs='?', type=float, default=DatasetLoader.DEFAULT_EXTRACTED_FOREST_SIZE_STOP, help='Specify the upper bound of the extracted forest sizes linspace.')
    parser.add_argument('--models_dir', nargs='?', type=str, default=DEFAULT_MODELS_DIR, help='The output directory of the trained models.')
    parser.add_argument('--dev_size', nargs='?', type=float, default=DatasetLoader.DEFAULT_DEV_SIZE, help='Dev subset ratio.')
    parser.add_argument('--test_size', nargs='?', type=float, default=DatasetLoader.DEFAULT_TEST_SIZE, help='Test subset ratio.')
    parser.add_argument('--random_seed_number', nargs='?', type=int, default=DatasetLoader.DEFAULT_RANDOM_SEED_NUMBER, help='Number of random seeds used.')
    parser.add_argument('--seeds', nargs='+', type=int, default=None, help='Specific a list of seeds instead of generate them randomly')
    parser.add_argument('--subsets_used', nargs='?', type=str, default=DatasetLoader.DEFAULT_SUBSETS_USED, help='train,dev: forest on train, OMP on dev. train+dev,train+dev: both forest and OMP on train+dev. train,train+dev: forest on train+dev and OMP on dev.')
    parser.add_argument('--normalize_weights', action='store_true', default=DatasetLoader.DEFAULT_NORMALIZE_WEIGHTS, help='Divide the predictions by the weights sum.')
    parser.add_argument('--verbose', action='store_true', default=DEFAULT_VERBOSE, help='Print tqdm progress bar.')
    parser.add_argument('--skip_best_hyperparams', action='store_true', default=DEFAULT_SKIP_BEST_HYPERPARAMS, help='Do not use the best hyperparameters if there exist.')
    parser.add_argument('--save_experiment_configuration', nargs='+', default=None, help='Save the experiment parameters specified in the command line in a file. Args: {{stage_num}} {{name}}')
    parser.add_argument('--job_number', nargs='?', type=int, default=DEFAULT_JOB_NUMBER, help='Specify the number of job used during the parallelisation across seeds.')
    parser.add_argument('--extraction_strategy', nargs='?', type=str, default=DEFAULT_EXTRACTION_STRATEGY, help='Specify the strategy to apply to extract the trees from the forest. Either omp, omp_nn, random, none, similarity_similarities, similarity_predictions, kmeans, ensemble.')
    parser.add_argument('--overwrite', action='store_true', default=DEFAULT_OVERWRITE, help='Overwrite the experiment id')
    args = parser.parse_args()

    if args.experiment_configuration:
        with open(args.experiment_configuration_path + os.sep + \
            args.experiment_configuration + '.json', 'r') as input_file:
            parameters = json.load(input_file)
    else:
        parameters = args.__dict__

    if parameters['extraction_strategy'] not in ['omp', 'omp_nn', 'omp_distillation', 'random', 'none', 'similarity_similarities', 'similarity_predictions', 'kmeans', 'ensemble']:
        raise ValueError('Specified extraction strategy {} is not supported.'.format(parameters['extraction_strategy']))

    pathlib.Path(parameters['models_dir']).mkdir(parents=True, exist_ok=True)

    logger = LoggerFactory.create(LOG_PATH, os.path.basename(__file__))

    hyperparameters_path = os.path.join('experiments', args.dataset_name, 'stage1', 'params.json')
    if os.path.exists(hyperparameters_path):
        logger.info("Hyperparameters found for this dataset at '{}'".format(hyperparameters_path))
        with open(hyperparameters_path, 'r+') as file_hyperparameter:
            loaded_hyperparameters = json.load(file_hyperparameter)['best_parameters']
            if args.skip_best_hyperparams:
                hyperparameters = {'n_estimators': loaded_hyperparameters['n_estimators']}
            else:
                hyperparameters = loaded_hyperparameters
    else:
        hyperparameters = {}

    """
    First case: no best hyperparameters are specified and no forest_size parameter
    specified in argument, so use the DEFAULT_FOREST_SIZE.
    Second case: no matter if hyperparameters are specified, the forest_size parameter
    will override it.
    Third implicit case: use the number of estimators found in specified hyperparameters.
    """
    if len(hyperparameters) == 0 and parameters['forest_size'] is None:
        hyperparameters['n_estimators'] = DatasetLoader.DEFAULT_FOREST_SIZE
    elif parameters['forest_size'] is not None:
        hyperparameters['n_estimators'] = parameters['forest_size']

    # The number of tree to extract from forest (K)
    parameters['extracted_forest_size'] = np.unique(np.around(hyperparameters['n_estimators'] *
        np.linspace(0, args.extracted_forest_size_stop,
        parameters['extracted_forest_size_samples'] + 1,
        endpoint=True)[1:]).astype(np.int)).tolist()

    logger.info(f"extracted forest sizes: {parameters['extracted_forest_size']}")

    if parameters['seeds'] != None and parameters['random_seed_number'] > 1:
        logger.warning('seeds and random_seed_number parameters are both specified. Seeds will be used.')    

    # Seeds are either provided as parameters or generated at random
    seeds = parameters['seeds'] if parameters['seeds'] is not None \
        else [random.randint(begin_random_seed_range, end_random_seed_range) \
        for i in range(parameters['random_seed_number'])]

    if args.experiment_id:
        experiment_id = args.experiment_id
        if args.overwrite:
            shutil.rmtree(os.path.join(parameters['models_dir'], str(experiment_id)), ignore_errors=True)
    else:
        # Resolve the next experiment id number (last id + 1)
        experiment_id = resolve_experiment_id(parameters['models_dir'])
    logger.info('Experiment id: {}'.format(experiment_id))

    """
    If the experiment configuration isn't coming from
    an already existing file, save it to a json file to
    keep trace of it (either a specified path, either in 'unnamed' dir.).
    """
    if args.experiment_configuration is None:
        if args.save_experiment_configuration:
            if len(args.save_experiment_configuration) != 2:
                raise ValueError('save_experiment_configuration must have two parameters.')
            elif int(args.save_experiment_configuration[0]) not in list(range(1, 6)):
                raise ValueError('save_experiment_configuration first parameter must be a supported stage id (i.e. [1, 5]).')
            output_experiment_stage_path = os.path.join(args.experiment_configuration_path,
                args.dataset_name, 'stage' + args.save_experiment_configuration[0])
            pathlib.Path(output_experiment_stage_path).mkdir(parents=True, exist_ok=True)
            output_experiment_configuration_path = os.path.join(output_experiment_stage_path,
                args.save_experiment_configuration[1] + '.json')
        else:
            pathlib.Path(os.path.join(args.experiment_configuration_path, 'unnamed')).mkdir(parents=True, exist_ok=True)
            output_experiment_configuration_path = os.path.join(
                args.experiment_configuration_path, 'unnamed', 'unnamed_{}.json'.format(
                experiment_id))
        with open(output_experiment_configuration_path, 'w') as output_file:
            json.dump(
                parameters,
                output_file,
                indent=4
            )

    # Run as much job as there are seeds
    with tqdm_joblib(tqdm(total=len(seeds), disable=not args.verbose)) as seed_job_pb:
        Parallel(n_jobs=args.job_number)(delayed(seed_job)(seed_job_pb, seeds[i],
            parameters, experiment_id, hyperparameters, args.verbose) for i in range(len(seeds)))
