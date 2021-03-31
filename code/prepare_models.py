import pathlib
import glob2
import os
import shutil
from tqdm import tqdm


if __name__ == "__main__":
    models_source_path = 'models'
    models_destination_path = 'bolsonaro_models_29-03-20'
    datasets = ['boston', 'diabetes', 'linnerud', 'breast_cancer', 'california_housing', 'diamonds',
        'steel-plates', 'kr-vs-kp', 'kin8nm', 'spambase', 'gamma', 'lfw_pairs']

    datasets = ['california_housing', 'boston', 'diabetes', 'breast_cancer', 'diamonds', 'steel-plates']

    pathlib.Path(models_destination_path).mkdir(parents=True, exist_ok=True)

    with tqdm(datasets) as dataset_bar:
        for dataset in dataset_bar:
            dataset_bar.set_description(dataset)
            found_paths = glob2.glob(os.path.join(models_source_path, dataset, 'stage5_27-03-20',
                '**', 'model_raw_results.pickle'), recursive=True)
            #pathlib.Path(os.path.join(models_destination_path, dataset)).mkdir(parents=True, exist_ok=True)
            with tqdm(found_paths) as found_paths_bar:
                for path in found_paths_bar:
                    found_paths_bar.set_description(path)
                    new_path = path.replace(f'models/{dataset}/stage5_27-03-20/', '')
                    (new_path, filename) = os.path.split(new_path)
                    if int(new_path.split(os.sep)[0]) != 9:
                        found_paths_bar.update(1)
                        found_paths_bar.set_description('Skipping...')
                        continue
                    new_path = os.path.join(models_destination_path, dataset, new_path)
                    pathlib.Path(new_path).mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src=path, dst=os.path.join(new_path, filename))
                    found_paths_bar.update(1)
            dataset_bar.update(1)
