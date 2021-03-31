from bolsonaro.models.model_raw_results import ModelRawResults
from bolsonaro.visualization.plotter import Plotter
from bolsonaro import LOG_PATH
from bolsonaro.error_handling.logger_factory import LoggerFactory
from bolsonaro.data.dataset_parameters import DatasetParameters
from bolsonaro.data.dataset_loader import DatasetLoader
from bolsonaro.data.task import Task

import argparse
import pathlib
from dotenv import find_dotenv, load_dotenv
import os
import numpy as np
import pickle
from tqdm import tqdm
from scipy.stats import rankdata
from pyrsa.vis.colors import rdm_colormap
from pyrsa.rdm.calc import calc_rdm
from pyrsa.data.dataset import Dataset
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error, accuracy_score


def vect2triu(dsm_vect, dim=None):
    if not dim:
        # sqrt(X²) \simeq sqrt(X²-X) -> sqrt(X²) = ceil(sqrt(X²-X))
        dim = int(np.ceil(np.sqrt(dsm_vect.shape[1] * 2)))
    dsm = np.zeros((dim,dim))
    ind_up = np.triu_indices(dim, 1)
    dsm[ind_up] = dsm_vect
    return dsm

def triu2full(dsm_triu):
    dsm_full = np.copy(dsm_triu)
    ind_low = np.tril_indices(dsm_full.shape[0], -1)
    dsm_full[ind_low] = dsm_full.T[ind_low]
    return dsm_full

def plot_RDM(rdm, file_path, condition_number):
    rdm = triu2full(vect2triu(rdm, condition_number))
    fig = plt.figure()
    cols = rdm_colormap(condition_number)
    plt.imshow(rdm, cmap=cols)
    plt.colorbar()
    plt.savefig(file_path, dpi=200)
    plt.close()



def retreive_extracted_forest_sizes_number(models_dir, experiment_id):
    experiment_id_path = models_dir + os.sep + str(experiment_id) # models/{experiment_id}
    experiment_seed_root_path = experiment_id_path + os.sep + 'seeds' # models/{experiment_id}/seeds
    seed = os.listdir(experiment_seed_root_path)[0]
    experiment_seed_path = experiment_seed_root_path + os.sep + seed # models/{experiment_id}/seeds/{seed}
    extracted_forest_sizes_root_path = experiment_seed_path + os.sep + 'extracted_forest_sizes'
    return len(os.listdir(extracted_forest_sizes_root_path))

def extract_scores_across_seeds_and_extracted_forest_sizes(models_dir, results_dir, experiment_id, weights=True, extracted_forest_sizes=list()):
    experiment_id_path = models_dir + os.sep + str(experiment_id) # models/{experiment_id}
    experiment_seed_root_path = experiment_id_path + os.sep + 'seeds' # models/{experiment_id}/seeds

    """
    Dictionaries to temporarly store the scalar results with the following structure:
    {seed_1: [score_1, ..., score_m], ... seed_n: [score_1, ..., score_k]}
    """
    experiment_train_scores = dict()
    experiment_dev_scores = dict()
    experiment_test_scores = dict()
    all_extracted_forest_sizes = list()

    # Used to check if all losses were computed using the same metric (it should be the case)
    experiment_score_metrics = list()

    # For each seed results stored in models/{experiment_id}/seeds
    seeds = os.listdir(experiment_seed_root_path)
    seeds.sort(key=int)
    for seed in seeds:
        experiment_seed_path = experiment_seed_root_path + os.sep + seed # models/{experiment_id}/seeds/{seed}
        extracted_forest_sizes_root_path = experiment_seed_path + os.sep + 'extracted_forest_sizes' # models/{experiment_id}/seeds/{seed}/forest_size

        # {{seed}:[]}
        experiment_train_scores[seed] = list()
        experiment_dev_scores[seed] = list()
        experiment_test_scores[seed] = list()

        if len(extracted_forest_sizes) == 0:
            # List the forest sizes in models/{experiment_id}/seeds/{seed}/extracted_forest_sizes
            extracted_forest_sizes = os.listdir(extracted_forest_sizes_root_path)
            extracted_forest_sizes = [nb_tree for nb_tree in extracted_forest_sizes if not 'no_weights' in nb_tree ]
            extracted_forest_sizes.sort(key=int)
        all_extracted_forest_sizes.append(list(map(int, extracted_forest_sizes)))
        for extracted_forest_size in extracted_forest_sizes:
            # models/{experiment_id}/seeds/{seed}/extracted_forest_sizes/{extracted_forest_size}
            if weights:
                extracted_forest_size_path = extracted_forest_sizes_root_path + os.sep + extracted_forest_size
            else:
                extracted_forest_size_path = extracted_forest_sizes_root_path + os.sep + extracted_forest_size + '_no_weights'
            # Load models/{experiment_id}/seeds/{seed}/extracted_forest_sizes/{extracted_forest_size}/model_raw_results.pickle file
            model_raw_results = ModelRawResults.load(extracted_forest_size_path)
            # Save the scores
            experiment_train_scores[seed].append(model_raw_results.train_score)
            experiment_dev_scores[seed].append(model_raw_results.dev_score)
            experiment_test_scores[seed].append(model_raw_results.test_score)
            # Save the metric
            experiment_score_metrics.append(model_raw_results.score_metric)

    # Sanity checks
    if len(set(experiment_score_metrics)) > 1:
        raise ValueError("The metrics used to compute the scores aren't the sames across seeds.")
    if len(set([sum(extracted_forest_sizes) for extracted_forest_sizes in all_extracted_forest_sizes])) != 1:
        raise ValueError("The extracted forest sizes aren't the sames across seeds.")

    return experiment_train_scores, experiment_dev_scores, experiment_test_scores, \
        all_extracted_forest_sizes[0], experiment_score_metrics[0]

def extract_scores_across_seeds_and_forest_size(models_dir, results_dir, experiment_id, extracted_forest_sizes_number):
    experiment_id_path = models_dir + os.sep + str(experiment_id) # models/{experiment_id}
    experiment_seed_root_path = experiment_id_path + os.sep + 'seeds' # models/{experiment_id}/seeds

    """
    Dictionaries to temporarly store the scalar results with the following structure:
    {seed_1: [score_1, ..., score_m], ... seed_n: [score_1, ..., score_k]}
    """
    experiment_train_scores = dict()
    experiment_dev_scores = dict()
    experiment_test_scores = dict()

    # Used to check if all losses were computed using the same metric (it should be the case)
    experiment_score_metrics = list()

    # For each seed results stored in models/{experiment_id}/seeds
    seeds = os.listdir(experiment_seed_root_path)
    seeds.sort(key=int)
    for seed in seeds:
        experiment_seed_path = experiment_seed_root_path + os.sep + seed # models/{experiment_id}/seeds/{seed}
        forest_size_root_path = experiment_seed_path + os.sep + 'forest_size' # models/{experiment_id}/seeds/{seed}/forest_size

        # {{seed}:[]}
        experiment_train_scores[seed] = list()
        experiment_dev_scores[seed] = list()
        experiment_test_scores[seed] = list()

        forest_size = os.listdir(forest_size_root_path)[0]
        # models/{experiment_id}/seeds/{seed}/forest_size/{forest_size}
        forest_size_path = forest_size_root_path + os.sep + forest_size
        # Load models/{experiment_id}/seeds/{seed}/extracted_forest_sizes/{extracted_forest_size}/model_raw_results.pickle file
        model_raw_results = ModelRawResults.load(forest_size_path)
        for _ in range(extracted_forest_sizes_number):
            # Save the scores
            experiment_train_scores[seed].append(model_raw_results.train_score)
            experiment_dev_scores[seed].append(model_raw_results.dev_score)
            experiment_test_scores[seed].append(model_raw_results.test_score)
            # Save the metric
            experiment_score_metrics.append(model_raw_results.score_metric)

    if len(set(experiment_score_metrics)) > 1:
        raise ValueError("The metrics used to compute the scores aren't the same everytime")

    return experiment_train_scores, experiment_dev_scores, experiment_test_scores, experiment_score_metrics[0]

def extract_weights_across_seeds(models_dir, results_dir, experiment_id):
    experiment_id_path = models_dir + os.sep + str(experiment_id) # models/{experiment_id}
    experiment_seed_root_path = experiment_id_path + os.sep + 'seeds' # models/{experiment_id}/seeds
    experiment_weights = dict()

    # For each seed results stored in models/{experiment_id}/seeds
    seeds = os.listdir(experiment_seed_root_path)
    seeds.sort(key=int)
    for seed in seeds:
        experiment_seed_path = experiment_seed_root_path + os.sep + seed # models/{experiment_id}/seeds/{seed}
        extracted_forest_sizes_root_path = experiment_seed_path + os.sep + 'extracted_forest_sizes' # models/{experiment_id}/seeds/{seed}/forest_size

        # {{seed}:[]}
        experiment_weights[seed] = list()

        # List the forest sizes in models/{experiment_id}/seeds/{seed}/extracted_forest_sizes
        extracted_forest_sizes = os.listdir(extracted_forest_sizes_root_path)
        extracted_forest_sizes = [nb_tree for nb_tree in extracted_forest_sizes if not 'no_weights' in nb_tree ]
        extracted_forest_sizes.sort(key=int)
        for extracted_forest_size in extracted_forest_sizes:
            # models/{experiment_id}/seeds/{seed}/extracted_forest_sizes/{extracted_forest_size}
            extracted_forest_size_path = extracted_forest_sizes_root_path + os.sep + extracted_forest_size
            # Load models/{experiment_id}/seeds/{seed}/extracted_forest_sizes/{extracted_forest_size}/model_raw_results.pickle file
            model_raw_results = ModelRawResults.load(extracted_forest_size_path)
            # Save the weights
            experiment_weights[seed].append(model_raw_results.model_weights)

    return experiment_weights

def extract_correlations_across_seeds(models_dir, results_dir, experiment_id):
    experiment_id_path = models_dir + os.sep + str(experiment_id) # models/{experiment_id}
    experiment_seed_root_path = experiment_id_path + os.sep + 'seeds' # models/{experiment_id}/seeds
    experiment_correlations = dict()

    # For each seed results stored in models/{experiment_id}/seeds
    seeds = os.listdir(experiment_seed_root_path)
    seeds.sort(key=int)
    for seed in seeds:
        experiment_seed_path = experiment_seed_root_path + os.sep + seed # models/{experiment_id}/seeds/{seed}
        extracted_forest_sizes_root_path = experiment_seed_path + os.sep + 'extracted_forest_sizes' # models/{experiment_id}/seeds/{seed}/forest_size

        # {{seed}:[]}
        experiment_correlations[seed] = list()

        # List the forest sizes in models/{experiment_id}/seeds/{seed}/extracted_forest_sizes
        extracted_forest_sizes = os.listdir(extracted_forest_sizes_root_path)
        extracted_forest_sizes = [nb_tree for nb_tree in extracted_forest_sizes if not 'no_weights' in nb_tree ]
        extracted_forest_sizes.sort(key=int)
        for extracted_forest_size in extracted_forest_sizes:
            # models/{experiment_id}/seeds/{seed}/extracted_forest_sizes/{extracted_forest_size}
            extracted_forest_size_path = extracted_forest_sizes_root_path + os.sep + extracted_forest_size
            # Load models/{experiment_id}/seeds/{seed}/extracted_forest_sizes/{extracted_forest_size}/model_raw_results.pickle file
            model_raw_results = ModelRawResults.load(extracted_forest_size_path)
            experiment_correlations[seed].append(model_raw_results.correlation)

    return experiment_correlations

def extract_coherences_across_seeds(models_dir, results_dir, experiment_id):
    experiment_id_path = models_dir + os.sep + str(experiment_id) # models/{experiment_id}
    experiment_seed_root_path = experiment_id_path + os.sep + 'seeds' # models/{experiment_id}/seeds
    experiment_coherences = dict()

    # For each seed results stored in models/{experiment_id}/seeds
    seeds = os.listdir(experiment_seed_root_path)
    seeds.sort(key=int)
    for seed in seeds:
        experiment_seed_path = experiment_seed_root_path + os.sep + seed # models/{experiment_id}/seeds/{seed}
        extracted_forest_sizes_root_path = experiment_seed_path + os.sep + 'extracted_forest_sizes' # models/{experiment_id}/seeds/{seed}/forest_size

        # {{seed}:[]}
        experiment_coherences[seed] = list()

        # List the forest sizes in models/{experiment_id}/seeds/{seed}/extracted_forest_sizes
        extracted_forest_sizes = os.listdir(extracted_forest_sizes_root_path)
        extracted_forest_sizes = [nb_tree for nb_tree in extracted_forest_sizes if not 'no_weights' in nb_tree ]
        extracted_forest_sizes.sort(key=int)
        for extracted_forest_size in extracted_forest_sizes:
            # models/{experiment_id}/seeds/{seed}/extracted_forest_sizes/{extracted_forest_size}
            extracted_forest_size_path = extracted_forest_sizes_root_path + os.sep + extracted_forest_size
            # Load models/{experiment_id}/seeds/{seed}/extracted_forest_sizes/{extracted_forest_size}/model_raw_results.pickle file
            model_raw_results = ModelRawResults.load(extracted_forest_size_path)
            experiment_coherences[seed].append(model_raw_results.coherence)

    return experiment_coherences

def extract_selected_trees_scores_across_seeds(models_dir, results_dir, experiment_id, weighted=False):
    experiment_id_path = models_dir + os.sep + str(experiment_id) # models/{experiment_id}
    experiment_seed_root_path = experiment_id_path + os.sep + 'seeds' # models/{experiment_id}/seeds
    experiment_selected_trees_scores = dict()

    print(f'[extract_selected_trees_scores_across_seeds] experiment_id: {experiment_id}')

    # For each seed results stored in models/{experiment_id}/seeds
    seeds = os.listdir(experiment_seed_root_path)
    seeds.sort(key=int)
    with tqdm(seeds) as seed_bar:
        for seed in seed_bar:
            seed_bar.set_description(f'seed: {seed}')
            experiment_seed_path = experiment_seed_root_path + os.sep + seed # models/{experiment_id}/seeds/{seed}
            extracted_forest_sizes_root_path = experiment_seed_path + os.sep + 'extracted_forest_sizes' # models/{experiment_id}/seeds/{seed}/forest_size

            dataset_parameters = DatasetParameters.load(experiment_seed_path, experiment_id)
            dataset = DatasetLoader.load(dataset_parameters)

            # {{seed}:[]}
            experiment_selected_trees_scores[seed] = list()

            # List the forest sizes in models/{experiment_id}/seeds/{seed}/extracted_forest_sizes
            extracted_forest_sizes = os.listdir(extracted_forest_sizes_root_path)
            extracted_forest_sizes = [nb_tree for nb_tree in extracted_forest_sizes if not 'no_weights' in nb_tree]
            extracted_forest_sizes.sort(key=int)
            with tqdm(extracted_forest_sizes) as extracted_forest_size_bar:
                for extracted_forest_size in extracted_forest_size_bar:
                    # models/{experiment_id}/seeds/{seed}/extracted_forest_sizes/{extracted_forest_size}
                    extracted_forest_size_path = extracted_forest_sizes_root_path + os.sep + extracted_forest_size
                    selected_trees = None
                    with open(os.path.join(extracted_forest_size_path, 'selected_trees.pickle'), 'rb') as file:
                        selected_trees = pickle.load(file)
                    selected_trees_test_scores = np.array([tree.score(dataset.X_test, dataset.y_test) for tree in selected_trees])

                    if weighted:
                        model_raw_results = ModelRawResults.load(extracted_forest_size_path)
                        weights = model_raw_results.model_weights
                        if type(weights) != str:
                            weights = weights[weights != 0]
                            score = np.mean(np.square(selected_trees_test_scores * weights))
                        else:
                            score = np.mean(np.square(selected_trees_test_scores))
                    else:
                        score = np.mean(selected_trees_test_scores)
                    experiment_selected_trees_scores[seed].append(score)
                    extracted_forest_size_bar.set_description(f'extracted_forest_size: {extracted_forest_size} - test_score: {round(score, 2)}')
                    extracted_forest_size_bar.update(1)
            seed_bar.update(1)

    return experiment_selected_trees_scores

def extract_selected_trees_across_seeds(models_dir, results_dir, experiment_id):
    experiment_id_path = models_dir + os.sep + str(experiment_id) # models/{experiment_id}
    experiment_seed_root_path = experiment_id_path + os.sep + 'seeds' # models/{experiment_id}/seeds
    experiment_selected_trees = dict()

    # For each seed results stored in models/{experiment_id}/seeds
    seeds = os.listdir(experiment_seed_root_path)
    seeds.sort(key=int)
    with tqdm(seeds) as seed_bar:
        for seed in seed_bar:
            seed_bar.set_description(f'seed: {seed}')
            experiment_seed_path = experiment_seed_root_path + os.sep + seed # models/{experiment_id}/seeds/{seed}
            extracted_forest_sizes_root_path = experiment_seed_path + os.sep + 'extracted_forest_sizes' # models/{experiment_id}/seeds/{seed}/forest_size

            dataset_parameters = DatasetParameters.load(experiment_seed_path, experiment_id)
            dataset = DatasetLoader.load(dataset_parameters)

            strength_metric = mean_squared_error if dataset.task == Task.REGRESSION \
                else lambda y_true, y_pred: accuracy_score(y_true, (y_pred -0.5)*2)

            X_train = np.concatenate([dataset.X_train, dataset.X_dev])
            y_train = np.concatenate([dataset.y_train, dataset.y_dev])    

            # {{seed}:[]}
            experiment_selected_trees[seed] = list()

            # List the forest sizes in models/{experiment_id}/seeds/{seed}/extracted_forest_sizes
            extracted_forest_sizes = os.listdir(extracted_forest_sizes_root_path)
            extracted_forest_sizes = [nb_tree for nb_tree in extracted_forest_sizes if not 'no_weights' in nb_tree ]
            extracted_forest_sizes.sort(key=int)
            all_selected_trees_predictions = list()
            with tqdm(extracted_forest_sizes) as extracted_forest_size_bar:
                for extracted_forest_size in extracted_forest_size_bar:
                    # models/{experiment_id}/seeds/{seed}/extracted_forest_sizes/{extracted_forest_size}
                    extracted_forest_size_path = extracted_forest_sizes_root_path + os.sep + extracted_forest_size
                    selected_trees = None
                    with open(os.path.join(extracted_forest_size_path, 'selected_trees.pickle'), 'rb') as file:
                        selected_trees = pickle.load(file)
                    selected_trees_train_scores = np.array([strength_metric(y_train, tree.predict(X_train)) for tree in selected_trees])
                    selected_trees_test_scores = np.array([strength_metric(dataset.y_test, tree.predict(dataset.X_test)) for tree in selected_trees])
                    train_strength = np.mean(selected_trees_train_scores)
                    test_strength = np.mean(selected_trees_test_scores)

                    model_raw_results_path = os.path.join(results_dir, str(experiment_id), 'seeds', str(seed), 'extracted_forest_sizes',
                        str(extracted_forest_size), 'model_raw_results.pickle')
                    with open(model_raw_results_path, 'rb') as file:
                        model_raw_results = pickle.load(file)
                    model_raw_results['train_scores'] = selected_trees_train_scores
                    model_raw_results['dev_scores'] = selected_trees_train_scores
                    model_raw_results['test_scores'] = selected_trees_test_scores
                    model_raw_results['train_strength'] = train_strength
                    model_raw_results['dev_strength'] = train_strength
                    model_raw_results['test_strength'] = test_strength
                    with open(model_raw_results_path, 'wb') as file:
                        pickle.dump(model_raw_results, file)

                    """#test_score = np.mean([tree.score(dataset.X_test, dataset.y_test) for tree in selected_trees])
                    #selected_trees_predictions = np.array([tree.score(dataset.X_test, dataset.y_test) for tree in selected_trees])
                    selected_trees_predictions = [tree.predict(dataset.X_test) for tree in selected_trees]
                    extracted_forest_size_bar.set_description(f'extracted_forest_size: {extracted_forest_size}')
                    #experiment_selected_trees[seed].append(test_score)
                    extracted_forest_size_bar.update(1)
                    selected_trees_predictions = np.array(selected_trees_predictions)
                    selected_trees_predictions = normalize(selected_trees_predictions)"""

                    """mds = MDS(len(selected_trees_predictions))
                    Y = mds.fit_transform(selected_trees_predictions)
                    plt.scatter(Y[:, 0], Y[:, 1])
                    plt.savefig(f'test_mds_{experiment_id}.png')"""

                    """if int(extracted_forest_size) <= 267:
                        forest_RDM = calc_rdm(Dataset(selected_trees_predictions), method='euclidean').get_vectors()
                        ranked_forest_RDM = np.apply_along_axis(rankdata, 1, forest_RDM.reshape(1, -1))

                        from scipy.cluster import hierarchy
                        RDM = triu2full(vect2triu(ranked_forest_RDM, int(extracted_forest_size)))
                        Z = hierarchy.linkage(RDM, 'average')
                        fig = plt.figure(figsize=(15, 8))
                        dn = hierarchy.dendrogram(Z)
                        plt.savefig(f'test_dendrogram_scores_id:{experiment_id}_seed:{seed}_size:{extracted_forest_size}.png')
                        plt.close()

                        plot_RDM(
                            rdm=ranked_forest_RDM,
                            file_path=f'test_scores_ranked_forest_RDM_id:{experiment_id}_seed:{seed}_size:{extracted_forest_size}.png',
                            condition_number=len(selected_trees_predictions)
                        )"""

            seed_bar.update(1)
    return experiment_selected_trees

if __name__ == "__main__":
    # get environment variables in .env
    load_dotenv(find_dotenv('.env'))

    DEFAULT_RESULTS_DIR = os.environ["project_dir"] + os.sep + 'results'
    DEFAULT_MODELS_DIR = os.environ["project_dir"] + os.sep + 'models'
    DEFAULT_PLOT_WEIGHT_DENSITY = False
    DEFAULT_WO_LOSS_PLOTS = False
    DEFAULT_PLOT_PREDS_COHERENCE = False
    DEFAULT_PLOT_FOREST_STRENGTH = False
    DEFAULT_COMPUTE_SELECTED_TREES_RDMS = False

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', nargs='?', type=int, required=True, help='Specify the stage number among [1, 5].')
    parser.add_argument('--experiment_ids', nargs='+', type=str, required=True, help='Compute the results of the specified experiment id(s).' + \
        'stage=1: {{base_with_params}} {{random_with_params}} {{omp_with_params}} {{base_wo_params}} {{random_wo_params}} {{omp_wo_params}}' + \
        'stage=2: {{no_normalization}} {{normalize_D}} {{normalize_weights}} {{normalize_D_and_weights}}' + \
        'stage=3: {{train-dev_subset}} {{train-dev_train-dev_subset}} {{train-train-dev_subset}}' + \
        'stage=5: {{base_with_params}} {{random_with_params}} {{omp_with_params}} [ensemble={{id}}] [similarity={{id}}] [kmean={{id}}]')
    parser.add_argument('--dataset_name', nargs='?', type=str, required=True, help='Specify the dataset name. TODO: read it from models dir directly.')
    parser.add_argument('--results_dir', nargs='?', type=str, default=DEFAULT_RESULTS_DIR, help='The output directory of the results.')
    parser.add_argument('--models_dir', nargs='?', type=str, default=DEFAULT_MODELS_DIR, help='The output directory of the trained models.')
    parser.add_argument('--plot_weight_density', action='store_true', default=DEFAULT_PLOT_WEIGHT_DENSITY, help='Plot the weight density. Only working for regressor models for now.')
    parser.add_argument('--wo_loss_plots', action='store_true', default=DEFAULT_WO_LOSS_PLOTS, help='Do not compute the loss plots.')
    parser.add_argument('--plot_preds_coherence', action='store_true', default=DEFAULT_PLOT_PREDS_COHERENCE, help='Plot the coherence of the prediction trees.')
    parser.add_argument('--plot_preds_correlation', action='store_true', default=DEFAULT_PLOT_PREDS_COHERENCE, help='Plot the correlation of the prediction trees.')
    parser.add_argument('--plot_forest_strength', action='store_true', default=DEFAULT_PLOT_FOREST_STRENGTH, help='Plot the strength of the extracted forest.')
    parser.add_argument('--compute_selected_trees_rdms', action='store_true', default=DEFAULT_COMPUTE_SELECTED_TREES_RDMS, help='Representation similarity analysis of the selected trees')
    args = parser.parse_args()

    if args.stage not in list(range(1, 6)):
        raise ValueError('stage must be a supported stage id (i.e. [1, 5]).')

    logger = LoggerFactory.create(LOG_PATH, os.path.basename(__file__))

    logger.info('Compute results of with stage:{} - experiment_ids:{} - dataset_name:{} - results_dir:{} - models_dir:{}'.format(
        args.stage, args.experiment_ids, args.dataset_name, args.results_dir, args.models_dir))

    # Create recursively the results dir tree
    pathlib.Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    if args.stage == 1 and not args.wo_loss_plots:
        if len(args.experiment_ids) != 6:
            raise ValueError('In the case of stage 1, the number of specified experiment ids must be 6.')

        # Retreive the extracted forest sizes number used in order to have a base forest axis as long as necessary
        extracted_forest_sizes_number = retreive_extracted_forest_sizes_number(args.models_dir, int(args.experiment_ids[1]))

        # Experiments that used the best hyperparameters found for this dataset

        # base_with_params
        logger.info('Loading base_with_params experiment scores...')
        base_with_params_train_scores, base_with_params_dev_scores, base_with_params_test_scores, \
            base_with_params_experiment_score_metric = \
            extract_scores_across_seeds_and_forest_size(args.models_dir, args.results_dir, int(args.experiment_ids[0]),
            extracted_forest_sizes_number)
        # random_with_params
        logger.info('Loading random_with_params experiment scores...')
        random_with_params_train_scores, random_with_params_dev_scores, random_with_params_test_scores, \
            with_params_extracted_forest_sizes, random_with_params_experiment_score_metric = \
            extract_scores_across_seeds_and_extracted_forest_sizes(args.models_dir, args.results_dir, int(args.experiment_ids[1]))
        # omp_with_params
        logger.info('Loading omp_with_params experiment scores...')
        omp_with_params_train_scores, omp_with_params_dev_scores, omp_with_params_test_scores, _, \
            omp_with_params_experiment_score_metric = extract_scores_across_seeds_and_extracted_forest_sizes(
                args.models_dir, args.results_dir, int(args.experiment_ids[2]))

        # Experiments that didn't use the best hyperparameters found for this dataset

        # base_wo_params
        logger.info('Loading base_wo_params experiment scores...')
        base_wo_params_train_scores, base_wo_params_dev_scores, base_wo_params_test_scores, \
            base_wo_params_experiment_score_metric = extract_scores_across_seeds_and_forest_size(
                args.models_dir, args.results_dir, int(args.experiment_ids[3]),
            extracted_forest_sizes_number)
        # random_wo_params
        logger.info('Loading random_wo_params experiment scores...')
        random_wo_params_train_scores, random_wo_params_dev_scores, random_wo_params_test_scores, \
            wo_params_extracted_forest_sizes, random_wo_params_experiment_score_metric = \
                extract_scores_across_seeds_and_extracted_forest_sizes(
                args.models_dir, args.results_dir, int(args.experiment_ids[4]))
        # omp_wo_params
        logger.info('Loading omp_wo_params experiment scores...')
        omp_wo_params_train_scores, omp_wo_params_dev_scores, omp_wo_params_test_scores, _, \
            omp_wo_params_experiment_score_metric = extract_scores_across_seeds_and_extracted_forest_sizes(
                args.models_dir, args.results_dir, int(args.experiment_ids[5]))

        # Sanity check on the metrics retreived
        if not (base_with_params_experiment_score_metric == random_with_params_experiment_score_metric ==
            omp_with_params_experiment_score_metric == base_wo_params_experiment_score_metric ==
            random_wo_params_experiment_score_metric ==
            omp_wo_params_experiment_score_metric):
            raise ValueError('Score metrics of all experiments must be the same.')
        experiments_score_metric = base_with_params_experiment_score_metric

        output_path = os.path.join(args.results_dir, args.dataset_name, 'stage1')
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        """all_experiment_scores_with_params=[base_with_params_train_scores, base_with_params_dev_scores, base_with_params_test_scores,
                random_with_params_train_scores, random_with_params_dev_scores, random_with_params_test_scores,
                omp_with_params_train_scores, omp_with_params_dev_scores, omp_with_params_test_scores],
            all_experiment_scores_wo_params=[base_wo_params_train_scores, base_wo_params_dev_scores, base_wo_params_test_scores,
                random_wo_params_train_scores, random_wo_params_dev_scores, random_wo_params_test_scores,
                omp_wo_params_train_scores, omp_wo_params_dev_scores, omp_wo_params_test_scores],
            all_labels=['base_with_params_train', 'base_with_params_dev', 'base_with_params_test',
                'random_with_params_train', 'random_with_params_dev', 'random_with_params_test',
                'omp_with_params_train', 'omp_with_params_dev', 'omp_with_params_test'],"""

        Plotter.plot_stage1_losses(
            file_path=output_path + os.sep + 'losses.png',
            all_experiment_scores_with_params=[base_with_params_test_scores,
                random_with_params_test_scores,
                omp_with_params_test_scores],
            all_experiment_scores_wo_params=[base_wo_params_test_scores,
                random_wo_params_test_scores,
                omp_wo_params_test_scores],
            all_labels=['base', 'random', 'omp'],
            x_value=with_params_extracted_forest_sizes,
            xlabel='Number of trees extracted',
            ylabel=experiments_score_metric,
            title='Loss values of {}\nusing best and default hyperparameters'.format(args.dataset_name)
        )
    elif args.stage == 2 and not args.wo_loss_plots:
        if len(args.experiment_ids) != 4:
            raise ValueError('In the case of stage 2, the number of specified experiment ids must be 4.')

        # no_normalization
        logger.info('Loading no_normalization experiment scores...')
        _, _, no_normalization_test_scores, extracted_forest_sizes, no_normalization_experiment_score_metric = \
            extract_scores_across_seeds_and_extracted_forest_sizes(args.models_dir, args.results_dir,
            int(args.experiment_ids[0]))

        # normalize_D
        logger.info('Loading normalize_D experiment scores...')
        _, _, normalize_D_test_scores, _, normalize_D_experiment_score_metric = \
            extract_scores_across_seeds_and_extracted_forest_sizes(args.models_dir, args.results_dir,
            int(args.experiment_ids[1]))

        # normalize_weights
        logger.info('Loading normalize_weights experiment scores...')
        _, _, normalize_weights_test_scores, _, normalize_weights_experiment_score_metric = \
            extract_scores_across_seeds_and_extracted_forest_sizes(args.models_dir, args.results_dir,
            int(args.experiment_ids[2]))

        # normalize_D_and_weights
        logger.info('Loading normalize_D_and_weights experiment scores...')
        _, _, normalize_D_and_weights_test_scores, _, normalize_D_and_weights_experiment_score_metric = \
            extract_scores_across_seeds_and_extracted_forest_sizes(args.models_dir, args.results_dir,
            int(args.experiment_ids[3]))

        # Sanity check on the metrics retreived
        if not (no_normalization_experiment_score_metric == normalize_D_experiment_score_metric
            == normalize_weights_experiment_score_metric == normalize_D_and_weights_experiment_score_metric):
            raise ValueError('Score metrics of all experiments must be the same.')
        experiments_score_metric = no_normalization_experiment_score_metric

        output_path = os.path.join(args.results_dir, args.dataset_name, 'stage2')
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        Plotter.plot_stage2_losses(
            file_path=output_path + os.sep + 'losses.png',
            all_experiment_scores=[no_normalization_test_scores, normalize_D_test_scores,
                normalize_weights_test_scores, normalize_D_and_weights_test_scores],
            all_labels=['no_normalization', 'normalize_D', 'normalize_weights', 'normalize_D_and_weights'],
            x_value=extracted_forest_sizes,
            xlabel='Number of trees extracted',
            ylabel=experiments_score_metric,
            title='Loss values of {}\nusing different normalizations'.format(args.dataset_name))
    elif args.stage == 3 and not args.wo_loss_plots:
        if len(args.experiment_ids) != 3:
            raise ValueError('In the case of stage 3, the number of specified experiment ids must be 3.')

        # train-dev_subset
        logger.info('Loading train-dev_subset experiment scores...')
        train_dev_subset_train_scores, train_dev_subset_dev_scores, train_dev_subset_test_scores, \
            extracted_forest_sizes, train_dev_subset_experiment_score_metric = \
            extract_scores_across_seeds_and_extracted_forest_sizes(args.models_dir, args.results_dir,
            int(args.experiment_ids[0]))

        # train-dev_train-dev_subset
        logger.info('Loading train-dev_train-dev_subset experiment scores...')
        train_dev_train_dev_subset_train_scores, train_dev_train_dev_subset_dev_scores, train_dev_train_dev_subset_test_scores, \
            _, train_dev_train_dev_subset_experiment_score_metric = \
            extract_scores_across_seeds_and_extracted_forest_sizes(args.models_dir, args.results_dir,
            int(args.experiment_ids[1]))

        # train-train-dev_subset
        logger.info('Loading train-train-dev_subset experiment scores...')
        train_train_dev_subset_train_scores, train_train_dev_subset_dev_scores, train_train_dev_subset_test_scores, \
            _, train_train_dev_subset_experiment_score_metric = \
            extract_scores_across_seeds_and_extracted_forest_sizes(args.models_dir, args.results_dir,
            int(args.experiment_ids[2]))

        # Sanity check on the metrics retreived
        if not (train_dev_subset_experiment_score_metric == train_dev_train_dev_subset_experiment_score_metric
            == train_train_dev_subset_experiment_score_metric):
            raise ValueError('Score metrics of all experiments must be the same.')
        experiments_score_metric = train_dev_subset_experiment_score_metric

        output_path = os.path.join(args.results_dir, args.dataset_name, 'stage3')
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        Plotter.plot_stage2_losses(
            file_path=output_path + os.sep + 'losses.png',
            all_experiment_scores=[train_dev_subset_test_scores, train_dev_train_dev_subset_test_scores,
                train_train_dev_subset_test_scores],
            all_labels=['train,dev', 'train+dev,train+dev', 'train,train+dev'],
            x_value=extracted_forest_sizes,
            xlabel='Number of trees extracted',
            ylabel=experiments_score_metric,
            title='Loss values of {}\nusing different training subsets'.format(args.dataset_name))

        """Plotter.plot_stage2_losses(
            file_path=output_path + os.sep + 'losses.png',
            all_experiment_scores=[train_dev_subset_train_scores, train_train_dev_subset_train_scores,
                train_train_dev_subset_train_scores, train_dev_subset_dev_scores, train_dev_train_dev_subset_dev_scores,
                train_train_dev_subset_dev_scores, train_dev_subset_test_scores, train_dev_train_dev_subset_test_scores,
                train_train_dev_subset_test_scores],
            all_labels=['train,dev - train', 'train+dev,train+dev - train', 'train,train+dev - train',
                'train,dev - dev', 'train+dev,train+dev - dev', 'train,train+dev - dev',
                'train,dev - test', 'train+dev,train+dev - test', 'train,train+dev - test'],
            x_value=extracted_forest_sizes,
            xlabel='Number of trees extracted',
            ylabel=experiments_score_metric,
            title='Loss values of {}\nusing different training subsets'.format(args.dataset_name))"""
    elif args.stage == 4 and not args.wo_loss_plots:
        if len(args.experiment_ids) != 3:
            raise ValueError('In the case of stage 4, the number of specified experiment ids must be 3.')

        # Retreive the extracted forest sizes number used in order to have a base forest axis as long as necessary
        extracted_forest_sizes_number = retreive_extracted_forest_sizes_number(args.models_dir, args.experiment_ids[1])

        # base_with_params
        logger.info('Loading base_with_params experiment scores...')
        base_with_params_train_scores, base_with_params_dev_scores, base_with_params_test_scores, \
            base_with_params_experiment_score_metric = \
            extract_scores_across_seeds_and_forest_size(args.models_dir, args.results_dir, int(args.experiment_ids[0]),
            extracted_forest_sizes_number)
        # random_with_params
        logger.info('Loading random_with_params experiment scores...')
        random_with_params_train_scores, random_with_params_dev_scores, random_with_params_test_scores, \
            with_params_extracted_forest_sizes, random_with_params_experiment_score_metric = \
            extract_scores_across_seeds_and_extracted_forest_sizes(args.models_dir, args.results_dir, int(args.experiment_ids[1]))
        # omp_with_params
        logger.info('Loading omp_with_params experiment scores...')
        """omp_with_params_train_scores, omp_with_params_dev_scores, omp_with_params_test_scores, _, \
            omp_with_params_experiment_score_metric, experiment_weights = extract_scores_across_seeds_and_extracted_forest_sizes(
                args.models_dir, args.results_dir, args.experiment_ids[2])"""
        omp_with_params_train_scores, omp_with_params_dev_scores, omp_with_params_test_scores, _, \
            omp_with_params_experiment_score_metric = extract_scores_across_seeds_and_extracted_forest_sizes(
                args.models_dir, args.results_dir, int(args.experiment_ids[2]))
        #omp_with_params_without_weights
        logger.info('Loading omp_with_params without weights experiment scores...')
        omp_with_params_without_weights_train_scores, omp_with_params_without_weights_dev_scores, omp_with_params_without_weights_test_scores, _, \
            omp_with_params_experiment_score_metric = extract_scores_across_seeds_and_extracted_forest_sizes(
                args.models_dir, args.results_dir, int(args.experiment_ids[2]), weights=False)

        """# base_with_params
        logger.info('Loading base_with_params experiment scores 2...')
        _, _, base_with_params_test_scores_2, \
            _ = \
            extract_scores_across_seeds_and_forest_size(args.models_dir, args.results_dir, args.experiment_ids[3],
            extracted_forest_sizes_number)
        # random_with_params
        logger.info('Loading random_with_params experiment scores 2...')
        _, _, random_with_params_test_scores_2, \
            _, _ = \
            extract_scores_across_seeds_and_extracted_forest_sizes(args.models_dir, args.results_dir, args.experiment_ids[4])"""

        # Sanity check on the metrics retreived
        if not (base_with_params_experiment_score_metric == random_with_params_experiment_score_metric
            == omp_with_params_experiment_score_metric):
            raise ValueError('Score metrics of all experiments must be the same.')
        experiments_score_metric = base_with_params_experiment_score_metric

        output_path = os.path.join(args.results_dir, args.dataset_name, 'stage4')
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        Plotter.plot_stage2_losses(
            file_path=output_path + os.sep + 'losses.png',
            all_experiment_scores=[base_with_params_test_scores, random_with_params_test_scores, omp_with_params_test_scores,
                                   omp_with_params_without_weights_test_scores],
            all_labels=['base', 'random', 'omp', 'omp_without_weights'],
            x_value=with_params_extracted_forest_sizes,
            xlabel='Number of trees extracted',
            ylabel=experiments_score_metric,
            title='Loss values of {}\nusing best params of previous stages'.format(args.dataset_name))
    elif args.stage == 5 and not args.wo_loss_plots:
        # Retreive the extracted forest sizes number used in order to have a base forest axis as long as necessary
        extracted_forest_sizes_number = retreive_extracted_forest_sizes_number(args.models_dir, int(args.experiment_ids[1]))
        all_labels = list()
        all_scores = list()

        """extracted_forest_sizes = np.unique(np.around(1000 *
            np.linspace(0, 1.0,
            30 + 1,
            endpoint=True)[1:]).astype(np.int)).tolist()"""

        #extracted_forest_sizes = [4, 7, 11, 14, 18, 22, 25, 29, 32, 36, 40, 43, 47, 50, 54, 58, 61, 65, 68, 72, 76, 79, 83, 86, 90, 94, 97, 101, 104, 108]

        #extracted_forest_sizes = [str(forest_size) for forest_size in extracted_forest_sizes]
        extracted_forest_sizes= list()

        # base_with_params
        logger.info('Loading base_with_params experiment scores...')
        base_with_params_train_scores, base_with_params_dev_scores, base_with_params_test_scores, \
            base_with_params_experiment_score_metric = \
            extract_scores_across_seeds_and_forest_size(args.models_dir, args.results_dir, int(args.experiment_ids[0]),
            extracted_forest_sizes_number)
        # random_with_params
        logger.info('Loading random_with_params experiment scores...')
        random_with_params_train_scores, random_with_params_dev_scores, random_with_params_test_scores, \
            with_params_extracted_forest_sizes, random_with_params_experiment_score_metric = \
            extract_scores_across_seeds_and_extracted_forest_sizes(args.models_dir, args.results_dir, int(args.experiment_ids[1]),
            extracted_forest_sizes=extracted_forest_sizes)
        # omp_with_params
        logger.info('Loading omp_with_params experiment scores...')
        omp_with_params_train_scores, omp_with_params_dev_scores, omp_with_params_test_scores, _, \
            omp_with_params_experiment_score_metric = extract_scores_across_seeds_and_extracted_forest_sizes(
                args.models_dir, args.results_dir, int(args.experiment_ids[2]), extracted_forest_sizes=extracted_forest_sizes)
        #omp_with_params_without_weights
        logger.info('Loading omp_with_params without weights experiment scores...')
        omp_with_params_without_weights_train_scores, omp_with_params_without_weights_dev_scores, omp_with_params_without_weights_test_scores, _, \
            omp_with_params_experiment_score_metric = extract_scores_across_seeds_and_extracted_forest_sizes(
                args.models_dir, args.results_dir, int(args.experiment_ids[2]), weights=False, extracted_forest_sizes=extracted_forest_sizes)

        """print(omp_with_params_dev_scores)
        import sys
        sys.exit(0)"""

        all_labels = ['base', 'random', 'omp', 'omp_wo_weights']
        #all_labels = ['base', 'random', 'omp']
        omp_with_params_test_scores_new = dict()
        filter_num = -1
        """filter_num = 9
        for key, value in omp_with_params_test_scores.items():
            omp_with_params_test_scores_new[key] = value[:filter_num]"""
        all_scores = [base_with_params_test_scores, random_with_params_test_scores, omp_with_params_test_scores,
            omp_with_params_without_weights_test_scores]
        #all_scores = [base_with_params_dev_scores, random_with_params_dev_scores, omp_with_params_dev_scores,
        #    omp_with_params_without_weights_dev_scores]
        #all_scores = [base_with_params_train_scores, random_with_params_train_scores, omp_with_params_train_scores,
        #    omp_with_params_without_weights_train_scores]

        for i in range(3, len(args.experiment_ids)):
            if 'kmeans' in args.experiment_ids[i]:
                label = 'kmeans'
            elif 'similarity_similarities' in args.experiment_ids[i]:
                label = 'similarity_similarities'
            elif 'similarity_predictions' in args.experiment_ids[i]:
                label = 'similarity_predictions'
            elif 'ensemble' in args.experiment_ids[i]:
                label = 'ensemble'
            elif 'omp_distillation' in args.experiment_ids[i]:
                label = 'omp_distillation'
            else:
                logger.error('Invalid value encountered')
                continue

            logger.info(f'Loading {label} experiment scores...')
            current_experiment_id = int(args.experiment_ids[i].split('=')[1])
            current_train_scores, current_dev_scores, current_test_scores, _, _ = extract_scores_across_seeds_and_extracted_forest_sizes(
                args.models_dir, args.results_dir, current_experiment_id)
            all_labels.append(label)
            all_scores.append(current_test_scores)
            #all_scores.append(current_train_scores)
            #all_scores.append(current_dev_scores)

        output_path = os.path.join(args.results_dir, args.dataset_name, 'stage5_test_train,dev')
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        Plotter.plot_stage2_losses(
            file_path=output_path + os.sep + f"losses_{'-'.join(all_labels)}_test_train,dev.png",
            all_experiment_scores=all_scores,
            all_labels=all_labels,
            x_value=with_params_extracted_forest_sizes,
            xlabel='Number of trees extracted',
            ylabel=base_with_params_experiment_score_metric,
            title='Loss values of {}\nusing best params of previous stages'.format(args.dataset_name), filter_num=filter_num)

    """if args.plot_weight_density:
        root_output_path = os.path.join(args.results_dir, args.dataset_name, f'stage{args.stage}')

        if args.stage == 1:
            omp_experiment_ids = [('omp_with_params', args.experiment_ids[2]), ('omp_wo_params', args.experiment_ids[2])]
        elif args.stage == 2:
            omp_experiment_ids = [('no_normalization', args.experiment_ids[0]),
                ('normalize_D', args.experiment_ids[1]),
                ('normalize_weights', args.experiment_ids[2]),
                ('normalize_D_and_weights', args.experiment_ids[3])]
        elif args.stage == 3:
            omp_experiment_ids = [('train-dev_subset', args.experiment_ids[0]),
                ('train-dev_train-dev_subset', args.experiment_ids[1]),
                ('train-train-dev_subset', args.experiment_ids[2])]
        elif args.stage == 4:
            omp_experiment_ids = [('omp_with_params', args.experiment_ids[2])]
        elif args.stage == 5:
            omp_experiment_ids = [('omp_with_params', args.experiment_ids[2])]
            for i in range(3, len(args.experiment_ids)):
                if 'kmeans' in args.experiment_ids[i]:
                    label = 'kmeans'
                elif 'similarity' in args.experiment_ids[i]:
                    label = 'similarity'
                elif 'ensemble' in args.experiment_ids[i]:
                    label = 'ensemble'
                else:
                    logger.error('Invalid value encountered')
                    continue

                current_experiment_id = int(args.experiment_ids[i].split('=')[1])
                omp_experiment_ids.append((label, current_experiment_id))

        for (experiment_label, experiment_id) in omp_experiment_ids:
            logger.info(f'Computing weight density plot for experiment {experiment_label}...')
            experiment_weights = extract_weights_across_seeds(args.models_dir, args.results_dir, experiment_id)
            Plotter.weight_density(experiment_weights, os.path.join(root_output_path, f'weight_density_{experiment_label}.png'))"""

    if args.plot_weight_density:
        logger.info(f'Computing weight density plot for experiment {experiment_label}...')
        experiment_weights = extract_weights_across_seeds(args.models_dir, args.results_dir, experiment_id)
        Plotter.weight_density(experiment_weights, os.path.join(root_output_path, f'weight_density_{experiment_label}.png'))
    if args.plot_preds_coherence:
        root_output_path = os.path.join(args.results_dir, args.dataset_name, f'stage5_new')
        pathlib.Path(root_output_path).mkdir(parents=True, exist_ok=True)
        all_labels = ['random', 'omp', 'kmeans', 'similarity_similarities', 'similarity_predictions', 'ensemble']
        _, _, _, with_params_extracted_forest_sizes, _ = \
            extract_scores_across_seeds_and_extracted_forest_sizes(args.models_dir, args.results_dir, 2)
        coherence_values = [extract_coherences_across_seeds(args.models_dir, args.results_dir, i) for i in args.experiment_ids]
        Plotter.plot_stage2_losses(
            file_path=root_output_path + os.sep + f"coherences_{'-'.join(all_labels)}.png",
            all_experiment_scores=coherence_values,
            all_labels=all_labels,
            x_value=with_params_extracted_forest_sizes,
            xlabel='Number of trees extracted',
            ylabel='Coherence',
            title='Coherence values of {}'.format(args.dataset_name))
        logger.info(f'Computing preds coherence plot...')

    if args.plot_preds_correlation:
        root_output_path = os.path.join(args.results_dir, args.dataset_name, f'stage5_new')
        pathlib.Path(root_output_path).mkdir(parents=True, exist_ok=True)
        all_labels = ['none', 'random', 'omp', 'kmeans', 'similarity_similarities', 'similarity_predictions', 'ensemble']
        _, _, _, with_params_extracted_forest_sizes, _ = \
            extract_scores_across_seeds_and_extracted_forest_sizes(args.models_dir, args.results_dir, 2)
        correlation_values = [extract_correlations_across_seeds(args.models_dir, args.results_dir, i) for i in args.experiment_ids]
        Plotter.plot_stage2_losses(
            file_path=root_output_path + os.sep + f"correlations_{'-'.join(all_labels)}.png",
            all_experiment_scores=correlation_values,
            all_labels=all_labels,
            x_value=with_params_extracted_forest_sizes,
            xlabel='Number of trees extracted',
            ylabel='correlation',
            title='correlation values of {}'.format(args.dataset_name))
        logger.info(f'Computing preds correlation plot...')

    if args.plot_forest_strength:
        root_output_path = os.path.join(args.results_dir, args.dataset_name, f'stage5_strength')
        pathlib.Path(root_output_path).mkdir(parents=True, exist_ok=True)

        _, _, _, with_params_extracted_forest_sizes, _ = \
                extract_scores_across_seeds_and_extracted_forest_sizes(args.models_dir, args.results_dir, 2)
        #all_selected_trees_scores = list()
        #all_selected_trees_weighted_scores = list()
        """with tqdm(args.experiment_ids) as experiment_id_bar:
            for experiment_id in experiment_id_bar:
                experiment_id_bar.set_description(f'experiment_id: {experiment_id}')
                selected_trees_scores, selected_trees_weighted_scores = extract_selected_trees_scores_across_seeds(
                    args.models_dir, args.results_dir, experiment_id)
                all_selected_trees_scores.append(selected_trees_scores)
                all_selected_trees_weighted_scores.append(selected_trees_weighted_scores)
                experiment_id_bar.update(1)"""

        #random_selected_trees_scores = extract_selected_trees_scores_across_seeds(
        #    args.models_dir, args.results_dir, 2, weighted=True)

        omp_selected_trees_scores = extract_selected_trees_scores_across_seeds(
            args.models_dir, args.results_dir, 3, weighted=True)

        similarity_similarities_selected_trees_scores = extract_selected_trees_scores_across_seeds(
            args.models_dir, args.results_dir, 6, weighted=True)

        #similarity_predictions_selected_trees_scores = extract_selected_trees_scores_across_seeds(
        #    args.models_dir, args.results_dir, 7)

        ensemble_selected_trees_scores = extract_selected_trees_scores_across_seeds(
            args.models_dir, args.results_dir, 8, weighted=True)

        # kmeans=5
        # similarity_similarities=6
        # similarity_predictions=7
        # ensemble=8

        all_selected_trees_scores = [random_selected_trees_scores, omp_selected_trees_scores, similarity_similarities_selected_trees_scores,
            ensemble_selected_trees_scores]

        with open('california_housing_forest_strength_scores.pickle', 'wb') as file:
            pickle.dump(all_selected_trees_scores, file)

        """with open('forest_strength_scores.pickle', 'rb') as file:
            all_selected_trees_scores = pickle.load(file)"""

        all_labels = ['random', 'omp', 'similarity_similarities', 'ensemble']

        Plotter.plot_stage2_losses(
            file_path=root_output_path + os.sep + f"forest_strength_{'-'.join(all_labels)}_v2_sota.png",
            all_experiment_scores=all_selected_trees_scores,
            all_labels=all_labels,
            x_value=with_params_extracted_forest_sizes,
            xlabel='Number of trees extracted',
            ylabel='Mean of selected tree scores on test set',
            title='Forest strength of {}'.format(args.dataset_name))

    if args.compute_selected_trees_rdms:
        root_output_path = os.path.join(args.results_dir, args.dataset_name, f'bolsonaro_models_29-03-20')
        #pathlib.Path(root_output_path).mkdir(parents=True, exist_ok=True)

        _, _, _, with_params_extracted_forest_sizes, _ = \
                extract_scores_across_seeds_and_extracted_forest_sizes(args.models_dir, args.results_dir, 2)
        all_selected_trees_scores = list()
        with tqdm(args.experiment_ids) as experiment_id_bar:
            for experiment_id in experiment_id_bar:
                experiment_id_bar.set_description(f'experiment_id: {experiment_id}')
                all_selected_trees_scores.append(extract_selected_trees_across_seeds(
                    args.models_dir, args.results_dir, experiment_id))
                experiment_id_bar.update(1)

        with open('forest_strength_scores.pickle', 'rb') as file:
            all_selected_trees_scores = pickle.load(file)

    logger.info('Done.')
