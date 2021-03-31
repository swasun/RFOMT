from pathlib import Path
import os
import pandas as pd
from pprint import pprint
import pickle
from collections import defaultdict
import numpy as np

from dotenv import load_dotenv, find_dotenv


dct_experiment_id_subset = dict((str(idx), "train+dev/train+dev") for idx in range(1, 10))
# dct_experiment_id_subset.update(dict((str(idx), "train/dev") for idx in range(9, 17)))

NONE = 'None'
Random = 'Random'
OMP = 'OMP'
OMPNN = 'NN-OMP'
OMP_Distillation = 'OMP Distillation'
Kmeans = 'Kmeans'
Zhang_Similarities = 'Zhang Similarities'
Zhang_Predictions = 'Zhang Predictions'
Ensemble = 'Ensemble'
dct_experiment_id_technique = {"1": NONE,
                               "2": Random,
                               "3": OMP,
                               "4": OMP_Distillation,
                               "5": Kmeans,
                               "6": Zhang_Similarities,
                               "7": Zhang_Predictions,
                               "8": Ensemble,
                               "9": OMPNN,
                               # "9": NONE,
                               # "10": Random,
                               # "11": OMP,
                               # "12": OMP_Distillation,
                               # "13": Kmeans,
                               # "14": Zhang_Similarities,
                               # "15": Zhang_Predictions,
                               # "16": Ensemble
                               }


dct_dataset_fancy = {
    "boston": "Boston",
    "breast_cancer": "Breast Cancer",
    "california_housing": "California Housing",
    "diabetes": "Diabetes",
    "diamonds": "Diamonds",
    "digits": "Digits",
    "iris": "Iris",
    "kin8nm": "Kin8nm",
    "kr-vs-kp": "KR-VS-KP",
    "olivetti_faces": "Olivetti Faces",
    "spambase": "Spambase",
    "steel-plates": "Steel Plates",
    "wine": "Wine",
    "gamma": "Gamma",
    "lfw_pairs": "LFW Pairs"
}

dct_dataset_base_forest_size = {
    "boston": 100,
    "breast_cancer": 1000,
    "california_housing": 1000,
    "diabetes": 108,
    "diamonds": 429,
    "digits": 1000,
    "iris": 1000,
    "kin8nm": 1000,
    "kr-vs-kp": 1000,
    "olivetti_faces": 1000,
    "spambase": 1000,
    "steel-plates": 1000,
    "wine": 1000,
    "gamma": 100,
    "lfw_pairs": 1000,
}

lst_attributes_tree_scores = ["dev_scores", "train_scores", "test_scores"]
skip_attributes = ["datetime"]

if __name__ == "__main__":

    load_dotenv(find_dotenv('.env'))
    # dir_name = "results/bolsonaro_models_25-03-20"
    # dir_name = "results/bolsonaro_models_27-03-20_v2"
    # dir_name = "results/bolsonaro_models_29-03-20"
    # dir_name = "results/bolsonaro_models_29-03-20_v3"
    # dir_name = "results/bolsonaro_models_29-03-20_v3"
    dir_name = "results/bolsonaro_models_29-03-20_v3_2"
    # dir_name = "results/bolsonaro_models_29-03-20"
    dir_path = Path(os.environ["project_dir"]) / dir_name

    output_dir_file = dir_path / "results.csv"

    dct_results = defaultdict(lambda: [])

    for root, dirs, files in os.walk(dir_path, topdown=False):
        for file_str in files:
            if file_str.split(".")[-1] != "pickle":
                continue
            # if file_str == "results.csv":
            #     continue
            path_dir = Path(root)
            path_file = path_dir / file_str
            print(path_file)
            try:
                with open(path_file, 'rb') as pickle_file:
                    obj_results = pickle.load(pickle_file)
            except:
                print("problem loading pickle file {}".format(path_file))

            path_dir_split = str(path_dir).split("/")

            bool_wo_weights = "no_weights" in str(path_file)

            if bool_wo_weights:
                forest_size = int(path_dir_split[-1].split("_")[0])
            else:
                forest_size = int(path_dir_split[-1])

            seed = int(path_dir_split[-3])
            id_xp = str(path_dir_split[-5])
            dataset = str(path_dir_split[-6])

            dct_results["forest_size"].append(forest_size)
            dct_results["seed"].append(seed)
            dct_results["dataset"].append(dct_dataset_fancy[dataset])
            dct_results["subset"].append(dct_experiment_id_subset[id_xp])
            dct_results["strategy"].append(dct_experiment_id_technique[id_xp])
            dct_results["wo_weights"].append(bool_wo_weights)
            dct_results["base_forest_size"].append(dct_dataset_base_forest_size[dataset])
            pruning_percent = forest_size / dct_dataset_base_forest_size[dataset]
            dct_results["pruning_percent"].append(np.round(pruning_percent, decimals=2))


            dct_nb_val_scores = {}
            nb_weights = None
            for key_result, val_result in obj_results.items():
                if key_result in skip_attributes:
                    continue

                #################################
                # Treat attribute model_weights #
                #################################
                if key_result == "model_weights":
                    if val_result == "":
                        dct_results["negative-percentage"].append(None)
                        dct_results["nb-non-zero-weight"].append(None)
                        nb_weights = None
                        continue
                    else:
                        lt_zero = val_result < 0
                        gt_zero = val_result > 0

                        nb_lt_zero = np.sum(lt_zero)
                        nb_gt_zero = np.sum(gt_zero)

                        percentage_lt_zero = nb_lt_zero / (nb_gt_zero + nb_lt_zero)
                        dct_results["negative-percentage"].append(percentage_lt_zero)

                        nb_weights = np.sum(val_result.astype(bool))
                        dct_results["nb-non-zero-weight"].append(nb_weights)
                        continue

                #####################
                # Treat tree scores #
                #####################
                if key_result in lst_attributes_tree_scores:
                    dct_nb_val_scores[key_result] = len(val_result)
                    continue

                if val_result == "":
                    val_result = None

                dct_results[key_result].append(val_result)

            assert all(key_scores in dct_nb_val_scores.keys() for key_scores in lst_attributes_tree_scores)
            len_scores = dct_nb_val_scores["test_scores"]
            assert all(dct_nb_val_scores[key_scores] == len_scores for key_scores in lst_attributes_tree_scores)
            dct_results["nb-scores"].append(len_scores)

            try:
                possible_actual_forest_size = (dct_results["forest_size"][-1], len_scores, nb_weights)
                min_forest_size = min(possible_actual_forest_size)
            except:
                possible_actual_forest_size = (dct_results["forest_size"][-1], len_scores)
                min_forest_size = min(possible_actual_forest_size)

            dct_results["actual-forest-size"].append(min_forest_size)


    final_df = pd.DataFrame.from_dict(dct_results)
    final_df.to_csv(output_dir_file)
    print(final_df)
