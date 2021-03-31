import os
import json
import pickle
from copy import deepcopy
import contextlib
import joblib

from sklearn.datasets import fetch_openml


def resolve_experiment_id(models_dir):
    """
    Return the ID of the next experiment.

    The ID is an int equal to n+1 where n is the current number of directory in `models_dir
    `
    :param models_dir:
    :return:
    """
    ids = [x for x in os.listdir(models_dir) 
        if os.path.isdir(models_dir + os.sep + x)]
    if len(ids) > 0:
        return int(max([int(i) for i in ids])) + 1
    return 1

def save_obj_to_json(file_path, attributes_dict):
    attributes = dict()
    for key, value in attributes_dict.items():
        attributes[key[1:]] = value
    with open(file_path, 'w') as output_file:
        json.dump(
            attributes,
            output_file,
            indent=4
        )

def load_obj_from_json(file_path, constructor):
    with open(file_path, 'r') as input_file:
        parameters = json.load(input_file)
    return constructor(**parameters)

def save_obj_to_pickle(file_path, attributes_dict):
    attributes = dict()
    for key, value in attributes_dict.items():
        attributes[key[1:]] = value
    with open(file_path, 'wb') as output_file:
        pickle.dump(attributes, output_file)

def load_obj_from_pickle(file_path, constructor):
    with open(file_path, 'rb') as input_file:
        parameters = pickle.load(input_file)
    return constructor(**parameters)

def binarize_class_data(data, class_pos, inplace=True):
    """
    Replace class_pos by +1 and ~class_pos by -1.

    :param data: an array of classes
    :param class_pos: the positive class to be replaced by +1
    :param inplace: If True, modify data in place (still return it, also)
    :return:
    """
    if not inplace:
        data = deepcopy(data)
    position_class_labels = (data == class_pos)
    data[~(position_class_labels)] = -1
    data[(position_class_labels)] = +1

    return data

def change_binary_func_load(base_load_function):
    def func_load(return_X_y, random_state=None):
        if random_state:
            X, y = base_load_function(return_X_y=return_X_y, random_state=random_state)
        else:
            X, y = base_load_function(return_X_y=return_X_y)
        possible_classes = sorted(set(y))
        assert len(possible_classes) == 2, "Function change binary_func_load only work for binary classfication"
        y = binarize_class_data(y, possible_classes[-1])
        return X, y
    return func_load

def change_binary_func_openml(dataset_name):
    def func_load(return_X_y=True, random_state=None):
        X, y = fetch_openml(dataset_name, return_X_y=return_X_y)
        possible_classes = sorted(set(y))
        assert len(possible_classes) == 2, "Function change binary_func_load only work for binary classfication"
        y = binarize_class_data(y, possible_classes[-1])
        y = y.astype('int')
        return X, y
    return func_load

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback:
        def __init__(self, time, index, parallel):
            self.index = index
            self.parallel = parallel

        def __call__(self, index):
            tqdm_object.update()
            if self.parallel._original_iterator is not None:
                self.parallel.dispatch_next()

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()    

def is_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

omp_premature_warning = """ Orthogonal matching pursuit ended prematurely due to linear
    dependence in the dictionary. The requested precision might not have been met.
    """
