import numpy as np

def score_metric_mse(y_preds, y_true):
    if len(y_true.shape) == 1:
        y_true = y_true[np.newaxis, :]
    if len(y_preds.shape) == 1:
        y_preds = y_preds[np.newaxis, :]
    assert y_preds.shape[1] == y_true.shape[1], "Number of examples to compare should be the same in y_preds and y_true"

    diff = y_preds - y_true
    squared_diff = diff ** 2
    mean_squared_diff = np.mean(squared_diff, axis=1)
    return mean_squared_diff

def score_metric_indicator(y_preds, y_true):
    if len(y_true.shape) == 1:
        y_true = y_true[np.newaxis, :]
    if len(y_preds.shape) == 1:
        y_preds = y_preds[np.newaxis, :]
    assert y_preds.shape[1] == y_true.shape[1], "Number of examples to compare should be the same in y_preds and y_true"

    bool_arr_correct_predictions = y_preds == y_true
    return np.average(bool_arr_correct_predictions, axis=1)

def aggregation_classification(predictions):
    return np.sign(np.sum(predictions, axis=0))

def aggregation_regression(predictions):
    return np.mean(predictions, axis=0)
